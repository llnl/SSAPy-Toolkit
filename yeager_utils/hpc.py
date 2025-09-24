# from mpi4py import MPI
import numpy as np


def get_unique_id2(rank, run_number, cpus_per_node):
    unique_id = int(np.arange(cpus_per_node * run_number, cpus_per_node * (run_number + 1))[rank])
    return unique_id


def get_unique_id(rank, run_number, cpus_per_node):
    """
    Calculates a unique ID based on the rank, run number, and CPUs per node.

    Parameters:
    rank (int): The rank of the CPU within the node.
    run_number (int): The current run number.
    cpus_per_node (int): The number of CPUs per node.

    Returns:
    int: The unique ID for the given rank, run number, and CPUs per node.
    """
    unique_id = rank + cpus_per_node * run_number
    return unique_id


def distribute_array_no_mpi(unique_id, total_jobs, array_size):
    # Calculate the base chunk size and remainder
    chunk_size = array_size // total_jobs
    remainder = array_size % total_jobs

    # Distribute the remainder: the first 'remainder' jobs get an extra element
    if unique_id < remainder:
        start_idx = unique_id * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = unique_id * chunk_size + remainder
        end_idx = start_idx + chunk_size

    # Handle cases where start_idx is out of bounds
    if start_idx >= array_size:
        return None, None  # No portion assigned for this unique_id
    
    return int(start_idx), int(end_idx)



# def distribute_array(array):
#     """
#     Distributes a 1D array among MPI processes.

#     Parameters:
#         array: numpy.ndarray
#             The 1D array to be distributed.
#         comm: MPI communicator
#             The MPI communicator.

#     Returns:
#         local_data: numpy.ndarray
#             The portion of array assigned to the current MPI process.
#     """
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     num_procs = comm.Get_size()

#     remainder = len(array) % num_procs
#     base_load = len(array) // num_procs
#     if rank == 0:
#         print('All processors will process at least {0} simulations.'.format(
#             base_load))
#         print('{0} processors will process an additional simulations'.format(
#             remainder))
#     load_list = np.concatenate((np.ones(remainder) * (base_load + 1),
#                                 np.ones(num_procs - remainder) * base_load))
#     if rank == 0:
#         print('load_list={0}'.format(load_list))
#     if rank < remainder:
#         local_array = np.zeros(base_load + 1, dtype=np.int64)
#     else:
#         local_array = np.zeros(base_load, dtype=np.int64)
#     disp = np.zeros(num_procs)
#     for i in range(len(load_list)):
#         if i == 0:
#             disp[i] = 0
#         else:
#             disp[i] = disp[i - 1] + load_list[i - 1]
#     comm.Scatterv([array, load_list, disp, MPI.DOUBLE], local_array)
#     print(f"Process {rank} received the indices {local_array}")
#     return local_array


# Example usage:
if __name__ == "__main__":
    array = np.arange(1000)  # or any list or numpy array
    for run_number in np.arange(10):
        for rank in np.arange(12):
            print("comparing unique id methods:", get_unique_id(rank, run_number, 12), get_unique_id2(rank, run_number, 12))
    # print(f"\nTesting distribute_array:\n")
    # local_data = distribute_array(array)
    # print(local_data)
    print(f"\nTesting distribute_array_no_mpi:\n")
    for unique_id in np.arange(10):
        start_idx, end_idx = distribute_array_no_mpi(unique_id, total_jobs=10, array_size=array.size)
        print(start_idx, end_idx)
        print(array[start_idx:end_idx])
        print()


# def mpi_scatter(scatter_array):
#     comm = MPI.COMM_WORLD  # Defines the default communicator
#     num_procs = comm.Get_size()  # Stores the number of processes in size.
#     rank = comm.Get_rank()  # Stores the rank (pid) of the current process
#     # stat = MPI.Status()
#     print(f'Number of procs: {num_procs}, rank: {rank}')
#     remainder = np.size(scatter_array) % num_procs
#     base_load = np.size(scatter_array) // num_procs
#     if rank == 0:
#         print('All processors will process at least {0} simulations.'.format(
#             base_load))
#         print('{0} processors will process an additional simulations'.format(
#             remainder))
#     load_list = np.concatenate((np.ones(remainder) * (base_load + 1),
#                                 np.ones(num_procs - remainder) * base_load))
#     if rank == 0:
#         print('load_list={0}'.format(load_list))
#     if rank < remainder:
#         scatter_array_local = np.zeros(base_load + 1, dtype=np.int64)
#     else:
#         scatter_array_local = np.zeros(base_load, dtype=np.int64)
#     disp = np.zeros(num_procs)
#     for i in range(np.size(load_list)):
#         if i == 0:
#             disp[i] = 0
#         else:
#             disp[i] = disp[i - 1] + load_list[i - 1]
#     comm.Scatterv([scatter_array, load_list, disp, MPI.DOUBLE], scatter_array_local)
#     print(f"Process {rank} received the scattered arrays: {scatter_array_local}")
#     return scatter_array_local, rank


# def mpi_scatter_exclude_rank_0(scatter_array):
#     # Function is for rank 0 to be used as a saving processor - all other processors will complete tasks.
#     comm = MPI.COMM_WORLD
#     num_procs = comm.Get_size()
#     rank = comm.Get_rank()
#     print(f'Number of procs: {num_procs}, rank: {rank}')

#     num_workers = num_procs - 1
#     remainder = np.size(scatter_array) % num_workers
#     base_load = np.size(scatter_array) // num_workers

#     if rank == 0:
#         print(f'All processors will process at least {base_load} simulations.')
#         print(f'{remainder} processors will process an additional simulation.')

#     load_list = np.concatenate((np.zeros(1), np.ones(remainder) * (base_load + 1),
#                                 np.ones(num_workers - remainder) * base_load))

#     if rank == 0:
#         print(f'load_list={load_list}')

#     scatter_array_local = np.zeros(int(load_list[rank]), dtype=np.int64)

#     disp = np.zeros(num_procs)
#     for i in range(1, num_procs):
#         disp[i] = disp[i - 1] + load_list[i - 1]

#     if rank == 0:
#         dummy_recvbuf = np.zeros(1, dtype=np.int64)
#         comm.Scatterv([scatter_array, load_list, disp, MPI.INT64_T], dummy_recvbuf)
#     else:
#         comm.Scatterv([scatter_array, load_list, disp, MPI.INT64_T], scatter_array_local)
#         print(f"Process {rank} received the {len(scatter_array_local)} element scattered array: {scatter_array_local}")

#     return scatter_array_local, rank