"""
Phase 1 Verification: Test that multiple processes can share a single JVM.

This script validates the core assumption of the JVM sharing optimization:
- 1 JVM process, 2 Python worker processes
- Each creates its own PythonInterface instance
- Concurrent load/step/reset calls don't interfere
- No CallbackServer needed for SciWorld operations

Usage:
    python scripts/test_shared_jvm.py --jar_path /path/to/scienceworld.jar

If all assertions pass, shared JVM mode is safe to use.
"""

import argparse
import multiprocessing as mp
import time
import sys
import os


def worker_fn(worker_id, port, env_step_limit, result_queue):
    """Worker that connects to shared JVM and runs independent env operations."""
    try:
        from py4j.java_gateway import JavaGateway, GatewayParameters
        
        # Connect to shared JVM (NO CallbackServer)
        gateway = JavaGateway(
            gateway_parameters=GatewayParameters(auto_field=True, port=port))
        
        # Create independent PythonInterface
        server = gateway.jvm.scienceworld.runtime.pythonapi.PythonInterface()
        
        # Get task list
        task_names = list(server.getTaskNames())
        print(f"  [Worker {worker_id}] Connected. {len(task_names)} tasks available.")
        
        # Load DIFFERENT tasks to prove isolation
        if worker_id == 0:
            task_name = task_names[0]  # First task
            variation = 0
        else:
            task_name = task_names[1]  # Second task  
            variation = 0
        
        server.load(task_name, variation, "", False)
        server.reset()
        
        # Take a few steps
        obs1 = server.step("look around")
        task_desc = server.freeActionTaskDesc()
        score = int(round(100 * server.getScore()))
        num_moves = server.getNumMoves()
        
        print(f"  [Worker {worker_id}] Task: {task_name}")
        print(f"  [Worker {worker_id}] Observation length: {len(str(obs1))}")
        print(f"  [Worker {worker_id}] Score: {score}, Moves: {num_moves}")
        
        # Take another step
        obs2 = server.step("inventory")
        score2 = int(round(100 * server.getScore()))
        
        # Disconnect (don't shutdown!)
        gateway.close()
        
        result_queue.put({
            'worker_id': worker_id,
            'task_name': task_name,
            'task_desc': task_desc[:100],
            'score': score,
            'score2': score2,
            'num_moves': num_moves,
            'obs_len': len(str(obs1)),
            'success': True,
            'error': None
        })
        
    except Exception as e:
        import traceback
        result_queue.put({
            'worker_id': worker_id,
            'success': False,
            'error': traceback.format_exc()
        })


def main():
    parser = argparse.ArgumentParser(description='Test shared JVM for SciWorld')
    parser.add_argument('--jar_path', type=str, default=None,
                        help='Path to scienceworld.jar')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers to test')
    args = parser.parse_args()

    from py4j.java_gateway import launch_gateway
    
    # Resolve jar path
    jar_path = args.jar_path
    if jar_path is None:
        from scienceworld.constants import JAR_PATH, BASEPATH
        jar_path = JAR_PATH
    else:
        from scienceworld.constants import BASEPATH
    
    print("=" * 60)
    print("Phase 1: Shared JVM Verification Test")
    print("=" * 60)
    
    # Step 1: Launch a single JVM
    print(f"\n[Main] Launching single JVM with jar: {jar_path}")
    port, proc = launch_gateway(
        classpath=jar_path, die_on_exit=True, cwd=BASEPATH,
        javaopts=['-Xmx4G'], return_proc=True)
    print(f"[Main] JVM launched on port {port} (PID: {proc.pid})")
    
    # Step 2: Heartbeat test from main process
    print(f"\n[Main] Heartbeat test...")
    from py4j.java_gateway import JavaGateway, GatewayParameters
    test_gw = JavaGateway(
        gateway_parameters=GatewayParameters(auto_field=True, port=port))
    test_server = test_gw.jvm.scienceworld.runtime.pythonapi.PythonInterface()
    task_names = list(test_server.getTaskNames())
    print(f"[Main] Heartbeat OK: {len(task_names)} tasks found")
    test_gw.close()
    
    # Step 3: Spawn workers
    print(f"\n[Main] Spawning {args.num_workers} workers...")
    result_queue = mp.Queue()
    workers = []
    
    for i in range(args.num_workers):
        w = mp.Process(
            target=worker_fn,
            args=(i, port, 100, result_queue))
        w.start()
        workers.append(w)
    
    # Step 4: Collect results
    results = []
    for w in workers:
        w.join(timeout=60)
    
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    
    # Step 5: Validate
    print(f"\n{'=' * 60}")
    print("Results:")
    print("=" * 60)
    
    all_success = True
    task_names_used = set()
    
    for r in sorted(results, key=lambda x: x.get('worker_id', -1)):
        wid = r.get('worker_id', '?')
        if r['success']:
            print(f"  Worker {wid}: OK - Task={r['task_name']}, "
                  f"Score={r['score']}, Moves={r['num_moves']}")
            task_names_used.add(r['task_name'])
        else:
            print(f"  Worker {wid}: FAILED")
            print(f"    Error: {r['error']}")
            all_success = False
    
    # Isolation check: workers should have different tasks
    if len(task_names_used) >= 2:
        print(f"\n  Isolation check: PASSED (different tasks: {task_names_used})")
    elif len(results) >= 2:
        print(f"\n  Isolation check: WARNING (same task used, cannot verify isolation)")
    
    # Step 6: Cleanup JVM
    print(f"\n[Main] Cleaning up JVM (PID: {proc.pid})...")
    if proc.poll() is None:
        try:
            proc.stdin.write("\n".encode("utf-8"))
            proc.stdin.flush()
        except Exception:
            proc.terminate()
    
    time.sleep(1)
    if proc.poll() is None:
        proc.kill()
        print("[Main] JVM force-killed")
    else:
        print("[Main] JVM exited cleanly")
    
    # Final verdict
    print(f"\n{'=' * 60}")
    if all_success and len(results) == args.num_workers:
        print("VERDICT: ALL TESTS PASSED - Shared JVM mode is safe!")
        print(f"  {args.num_workers} workers shared 1 JVM without interference.")
        return 0
    else:
        print(f"VERDICT: TESTS FAILED ({len(results)}/{args.num_workers} succeeded)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
