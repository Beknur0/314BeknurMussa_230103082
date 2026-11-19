import hashlib
import subprocess

from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15


def get_gpu_uuid():
    """Получить UUID активного GPU."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return output.strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    """Получить 64-битный seed из GitHub-логина и UUID GPU."""
    text = f"{github_user.strip()}::{gpu_uuid}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(f"Expected 4 MPI ranks, got {size}")

    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    cart_comm = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False,
    )

    cart_rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(cart_rank)

    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return {
        "cart_comm": cart_comm,
        "rank": cart_rank,
        "coords": coords,
        "neighbors": (nbr_up, nbr_down, nbr_left, nbr_right),
        "gpu_uuid": gpu_uuid,
        "instance_seed": instance_seed,
    }

import numpy as np


def initialize_subdomains(ctx):
    """Создать локальные u, kappa и source с ghost-слоем."""
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2
    row_rank, col_rank = ctx["coords"]

    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    # Только локальные координаты, включая ghost-слой.
    x = (col_rank * nx + np.arange(nx + 2)) * dx
    y = (row_rank * ny + np.arange(ny + 2)) * dy
    X, Y = np.meshgrid(x, y)

    # Диапазон: [0.08, 0.24].
    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)

    # Одинаковые параметры импульсов на всех рангах:
    # каждый вычисляет их на своих локальных координатах.
    rng = np.random.default_rng(ctx["instance_seed"])
    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)

    for _ in range(3):
        cx, cy = rng.uniform(0.25, 0.75, size=2)
        amplitude = rng.uniform(0.4, 1.0)
        width = rng.uniform(0.04, 0.08)

        u += amplitude * np.exp(
            -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * width ** 2)
        )

    # Локализованный источник малой интенсивности.
    source = 0.001 * np.exp(
        -((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2.0 * 0.05 ** 2)
    )

    # Физические границы Дирихле находятся в ghost-слое.
    up, down, left, right = ctx["neighbors"]
    if up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if down == MPI.PROC_NULL:
        u[-1, :] = 0.0
    if left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if right == MPI.PROC_NULL:
        u[:, -1] = 0.0

    return (
        np.ascontiguousarray(u),
        np.ascontiguousarray(kappa),
        np.ascontiguousarray(source),
        dx,
        dy,
    )

@cuda.jit
def pack_halos_kernel(u, up, down, left, right, ny, nx):
    t = cuda.grid(1)

    if t < nx:
        up[t] = u[1, t + 1]
        down[t] = u[ny, t + 1]

    if t < ny:
        left[t] = u[t + 1, 1]
        right[t] = u[t + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, up, down, left, right, ny, nx):
    t = cuda.grid(1)

    if t < nx:
        u[0, t + 1] = up[t]
        u[ny + 1, t + 1] = down[t]

    if t < ny:
        u[t + 1, 0] = left[t]
        u[t + 1, nx + 1] = right[t]

class HaloBuffers:
    def __init__(self, ny, nx):
        self.ny = ny
        self.nx = nx
        lengths = (nx, nx, ny, ny)

        self.device_send = [
            cuda.device_array(n, dtype=np.float64) for n in lengths
        ]
        self.device_recv = [
            cuda.device_array(n, dtype=np.float64) for n in lengths
        ]

        self.host_send = [
            cuda.pinned_array(n, dtype=np.float64) for n in lengths
        ]
        self.host_recv = [
            cuda.pinned_array(n, dtype=np.float64) for n in lengths
        ]

        for buf in self.host_recv:
            buf.fill(0.0)


def exchange_halos_async(u, ctx, buffers, stream_boundary):
    ny, nx = buffers.ny, buffers.nx
    threads = 128
    blocks = (max(nx, ny) + threads - 1) // threads

    pack_halos_kernel[blocks, threads, stream_boundary](
        u, *buffers.device_send, ny, nx
    )

    for device_buf, host_buf in zip(
        buffers.device_send, buffers.host_send
    ):
        device_buf.copy_to_host(host_buf, stream=stream_boundary)

    stream_boundary.synchronize()

    comm = ctx["cart_comm"]
    neighbors = ctx["neighbors"]

    # Direction order: up, down, left, right.
    send_tags = (0, 1, 2, 3)
    recv_tags = (1, 0, 3, 2)

    requests = []

    for neighbor, tag, host_buf in zip(
        neighbors, recv_tags, buffers.host_recv
    ):
        requests.append(comm.Irecv(host_buf, source=neighbor, tag=tag))

    for neighbor, tag, host_buf in zip(
        neighbors, send_tags, buffers.host_send
    ):
        requests.append(comm.Isend(host_buf, dest=neighbor, tag=tag))

    return requests


def finish_halo_exchange(u, buffers, requests, stream_boundary):
    MPI.Request.Waitall(requests)

    for host_buf, device_buf in zip(
        buffers.host_recv, buffers.device_recv
    ):
        device_buf.copy_to_device(host_buf, stream=stream_boundary)

    ny, nx = buffers.ny, buffers.nx
    threads = 128
    blocks = (max(nx, ny) + threads - 1) // threads

    unpack_halos_kernel[blocks, threads, stream_boundary](
        u, *buffers.device_recv, ny, nx
    )

@cuda.jit
def tiled_stencil_kernel(
    u, u_next, kappa, source,
    dt, dx, dy, damping,
    r_start, r_end, c_start, c_end,
):
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = r_start + cuda.blockIdx.y * TILE_DIM + ty
    col = c_start + cuda.blockIdx.x * TILE_DIM + tx

    sy = ty + 1
    sx = tx + 1

    height = u.shape[0]
    width = u.shape[1]

    if row <= r_end and col <= c_end and row < height and col < width:
        tile[sy, sx] = u[row, col]
    else:
        tile[sy, sx] = 0.0

    if ty == 0:
        if row > 0 and row - 1 < height and col <= c_end and col < width:
            tile[0, sx] = u[row - 1, col]
        else:
            tile[0, sx] = 0.0

    if ty == TILE_DIM - 1:
        if row + 1 <= r_end and row + 1 < height and col <= c_end and col < width:
            tile[SHARED_DIM - 1, sx] = u[row + 1, col]
        else:
            tile[SHARED_DIM - 1, sx] = 0.0

    if tx == 0:
        if col > 0 and col - 1 < width and row <= r_end and row < height:
            tile[sy, 0] = u[row, col - 1]
        else:
            tile[sy, 0] = 0.0

    if tx == TILE_DIM - 1:
        if col + 1 <= c_end and col + 1 < width and row <= r_end and row < height:
            tile[sy, SHARED_DIM - 1] = u[row, col + 1]
        else:
            tile[sy, SHARED_DIM - 1] = 0.0

    cuda.syncthreads()

    if row < r_end and col < c_end:
        center = tile[sy, sx]
        north = tile[sy - 1, sx]
        south = tile[sy + 1, sx]
        west = tile[sy, sx - 1]
        east = tile[sy, sx + 1]

        kc = kappa[row, col]
        kn = 0.5 * (kc + kappa[row - 1, col])
        ks = 0.5 * (kc + kappa[row + 1, col])
        kw = 0.5 * (kc + kappa[row, col - 1])
        ke = 0.5 * (kc + kappa[row, col + 1])

        diffusion_x = (ke * (east - center) - kw * (center - west)) / (dx * dx)
        diffusion_y = (ks * (south - center) - kn * (center - north)) / (dy * dy)

        u_next[row, col] = center + dt * (
            diffusion_x + diffusion_y
            - damping * center * center * center
            + source[row, col]
        )

def launch_stencil(
    u, u_next, kappa, source, dt, dx, dy,
    r_start, r_end, c_start, c_end, stream,
):
    blocks = (
        (c_end - c_start + TILE_DIM - 1) // TILE_DIM,
        (r_end - r_start + TILE_DIM - 1) // TILE_DIM,
    )

    tiled_stencil_kernel[blocks, (TILE_DIM, TILE_DIM), stream](
        u, u_next, kappa, source,
        dt, dx, dy, LAMBDA_DAMP,
        r_start, r_end, c_start, c_end,
    )


def step_overlapped(
    u, u_next, kappa, source, ctx, buffers,
    stream_boundary, stream_interior, dt, dx, dy,
):
    ny, nx = buffers.ny, buffers.nx

    requests = exchange_halos_async(u, ctx, buffers, stream_boundary)

    launch_stencil(
        u, u_next, kappa, source, dt, dx, dy,
        2, ny, 2, nx, stream_interior,
    )

    finish_halo_exchange(u, buffers, requests, stream_boundary)

    launch_stencil(
        u, u_next, kappa, source, dt, dx, dy,
        1, 2, 1, nx + 1, stream_boundary,
    )
    launch_stencil(
        u, u_next, kappa, source, dt, dx, dy,
        ny, ny + 1, 1, nx + 1, stream_boundary,
    )
    launch_stencil(
        u, u_next, kappa, source, dt, dx, dy,
        2, ny, 1, 2, stream_boundary,
    )
    launch_stencil(
        u, u_next, kappa, source, dt, dx, dy,
        2, ny, nx, nx + 1, stream_boundary,
    )

    stream_boundary.synchronize()
    stream_interior.synchronize()


def step_serialized(
    u, u_next, kappa, source, ctx, buffers,
    stream_boundary, dt, dx, dy,
):
    ny, nx = buffers.ny, buffers.nx

    requests = exchange_halos_async(u, ctx, buffers, stream_boundary)
    finish_halo_exchange(u, buffers, requests, stream_boundary)

    launch_stencil(
        u, u_next, kappa, source, dt, dx, dy,
        1, ny + 1, 1, nx + 1, stream_boundary,
    )

    stream_boundary.synchronize()

import json
import sys
import time


def global_metrics(cart_comm, current, previous, dx, dy):
    current_host = current.copy_to_host()[1:-1, 1:-1]

    local_energy = np.sum(current_host, dtype=np.float64) * dx * dy

    if previous is None:
        local_residual_sq = 0.0
    else:
        previous_host = previous.copy_to_host()[1:-1, 1:-1]
        difference = current_host - previous_host
        local_residual_sq = np.sum(difference * difference, dtype=np.float64)

    local_values = np.array(
        [local_energy, local_residual_sq],
        dtype=np.float64,
    )
    global_values = np.empty_like(local_values)

    cart_comm.Allreduce(local_values, global_values, op=MPI.SUM)

    return float(global_values[0]), float(np.sqrt(global_values[1]))


def inspect_stencil_ptx():
    asm = tiled_stencil_kernel.inspect_asm()
    ptx = "\n".join(asm.values()) if isinstance(asm, dict) else asm

    return {
        "st.shared.f64": ptx.count("st.shared.f64"),
        "ld.shared.f64": ptx.count("ld.shared.f64"),
        "bar.sync": ptx.count("bar.sync"),
    }


def run_mode(mode, ctx, initial_u, kappa, source, dt, dx, dy):
    cart_comm = ctx["cart_comm"]
    rank = ctx["rank"]
    ny = nx = 256

    d_kappa = cuda.to_device(kappa)
    d_source = cuda.to_device(source)
    buffers = HaloBuffers(ny, nx)
    boundary = cuda.stream()
    interior = cuda.stream()

    def execute_step(current, next_u):
        if mode == "overlapped":
            step_overlapped(
                current, next_u, d_kappa, d_source, ctx, buffers,
                boundary, interior, dt, dx, dy,
            )
        else:
            step_serialized(
                current, next_u, d_kappa, d_source, ctx, buffers,
                boundary, dt, dx, dy,
            )

    # Compile and warm up before measuring.
    warm_current = cuda.to_device(initial_u)
    warm_next = cuda.to_device(np.zeros_like(initial_u))
    execute_step(warm_current, warm_next)

    current = cuda.to_device(initial_u)
    next_u = cuda.to_device(np.zeros_like(initial_u))

    energy, residual = global_metrics(cart_comm, current, None, dx, dy)
    trace = [{"step": 0, "energy": energy, "l2_residual": residual}]

    if rank == 0:
        print(
            f"{mode} step=0 energy={energy:.15e} "
            f"l2_residual={residual:.15e}",
            flush=True,
        )

    start_event = cuda.event(timing=True)
    end_event = cuda.event(timing=True)

    cart_comm.Barrier()
    start_time = time.perf_counter()
    start_event.record(boundary)

    for step in range(1, STEPS + 1):
        execute_step(current, next_u)
        current, next_u = next_u, current

        if step % 100 == 0:
            energy, residual = global_metrics(
                cart_comm, current, next_u, dx, dy
            )
            trace.append({
                "step": step,
                "energy": energy,
                "l2_residual": residual,
            })

            if rank == 0:
                print(
                    f"{mode} step={step} energy={energy:.15e} "
                    f"l2_residual={residual:.15e}",
                    flush=True,
                )

    end_event.record(boundary)
    end_event.synchronize()
    local_wall_ms = (time.perf_counter() - start_time) * 1000.0
    local_event_ms = cuda.event_elapsed_time(start_event, end_event)

    wall_ms = cart_comm.allreduce(local_wall_ms, op=MPI.MAX)
    event_ms = cart_comm.allreduce(local_event_ms, op=MPI.MAX)

    return current.copy_to_host(), trace, wall_ms, event_ms


def run_solver(github_user):
    ctx = init_distributed_context(github_user)
    cart_comm = ctx["cart_comm"]
    rank = ctx["rank"]

    initial_u, kappa, source, dx, dy = initialize_subdomains(ctx)

    local_kappa_max = float(np.max(kappa))
    kappa_max = cart_comm.allreduce(local_kappa_max, op=MPI.MAX)
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = 0.5 * dt_max

    if rank == 0:
        print(f"gpu_uuid={ctx['gpu_uuid']}", flush=True)
        print(f"instance_seed={ctx['instance_seed']}", flush=True)
        print(f"mpi_ranks={cart_comm.Get_size()} dims=[2, 2]", flush=True)
        print(
            f"kappa_max={kappa_max:.15e} dx={dx:.15e} "
            f"dt_max={dt_max:.15e} dt={dt:.15e}",
            flush=True,
        )

    serialized, serial_trace, serialized_ms, serialized_event_ms = run_mode(
        "serialized", ctx, initial_u, kappa, source, dt, dx, dy
    )

    overlapped, overlap_trace, overlapped_ms, overlapped_event_ms = run_mode(
        "overlapped", ctx, initial_u, kappa, source, dt, dx, dy
    )

    local_diff = float(np.max(np.abs(
        serialized[1:-1, 1:-1] - overlapped[1:-1, 1:-1]
    )))
    parity_max_diff = cart_comm.allreduce(local_diff, op=MPI.MAX)

    ptx_counts = inspect_stencil_ptx()
    local_ptx_passed = all(count > 0 for count in ptx_counts.values())
    ptx_passed = cart_comm.allreduce(local_ptx_passed, op=MPI.LAND)

    if rank == 0:
        print(f"ptx_counts={ptx_counts}", flush=True)
        print(f"ptx_audit_passed={ptx_passed}", flush=True)
        print(f"parity_max_diff={parity_max_diff:.15e}", flush=True)
        print(
            f"serialized_ms={serialized_ms:.3f} "
            f"overlapped_ms={overlapped_ms:.3f}",
            flush=True,
        )
        print(
            f"serialized_cuda_event_ms={serialized_event_ms:.3f} "
            f"overlapped_cuda_event_ms={overlapped_event_ms:.3f}",
            flush=True,
        )

    if not ptx_passed:
        raise RuntimeError("PTX audit failed")
    if parity_max_diff >= 1e-12:
        raise RuntimeError("Serialized and overlapped results do not match")

    if rank == 0:
        manifest = {
            "github_username": github_user.strip(),
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": ctx["instance_seed"],
            "mpi_ranks": cart_comm.Get_size(),
            "process_grid": [2, 2],
            "steps": STEPS,
            "kappa_max": kappa_max,
            "dx": dx,
            "dy": dy,
            "dt": dt,
            "dt_max": dt_max,
            "serialized_ms": serialized_ms,
            "overlapped_ms": overlapped_ms,
            "serialized_cuda_event_ms": serialized_event_ms,
            "overlapped_cuda_event_ms": overlapped_event_ms,
            "parity_max_diff": parity_max_diff,
            "ptx_counts": ptx_counts,
            "ptx_audit_passed": bool(ptx_passed),
            "serialized_trace": serial_trace,
            "overlapped_trace": overlap_trace,
        }

        with open("submission_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("Wrote submission_manifest.json", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python distributed_cuda_solver.py GITHUB_USERNAME")
    run_solver(sys.argv[1])
