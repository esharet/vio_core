import vpi
import numpy as np

# Create a NumPy array of points
points = np.array([[100, 200], [150, 250], [300, 400]], dtype=np.float32)

    # Convert to a VPI Array
    vpi_points = vpi.asarray(points_np)

    print(f"Original number of points: {vpi_points.size}")

    print("\n--------- Method 1: Direct Indexing (Simple but less efficient) ---------")
    # You can access individual elements using standard Python indexing.
    # This is convenient for quick checks but can be slow if done in a loop.
    print(f"The first point is: {vpi_points[0]}")
    print(f"The second point is: {vpi_points[1]}")

    print("\n--------- Method 2: Locking (Recommended for performance) ---------")
    # The best practice is to lock the array's memory for CPU access.
    # This gives you a NumPy array that directly maps to the VPI array's data.

    # Use rlock_cpu() for read-only access
    print("Reading elements:")
    with vpi_points.rlock_cpu() as data:
        # 'data' is a NumPy array view of the VPI array's memory
        print(f"The underlying data type is: {type(data)}")
        print("All points:\n", data)
        # You can access elements like any NumPy array
        print(f"The third point is: {data[2]}")

    # Use lock_cpu() for read-write access
    print("\nModifying an element:")
    with vpi_points.lock_cpu() as data:
        print(f"Original first point: {data[0]}")
        # Modify the data in-place
        data[0] = [111, 222]
        print(f"Modified first point: {data[0]}")

    print("\nVerifying the change in the original VPI array:")
    # Accessing it again shows the change
    with vpi_points.rlock_cpu() as data:
        print("All points after modification:\n", data)

if __name__ == "__main__":
    main()
