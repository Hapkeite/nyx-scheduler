import torch
import torchvision.models as models
import time

def run_resnet_benchmark(num_batches=10, batch_size=32):
    print(f"--- Starting ResNet50 Benchmark ---")
    print(f"Batches: {num_batches}, Batch Size: {batch_size}")
    
    # 1. Start the timer. 
    # This is required to calculate your Job Completion Time (JCT) metric.
    start_time = time.time()

    # 2. Load the Model
    # We use a standard ResNet50. Moving it to ".cuda()" forces PyTorch to 
    # allocate memory on the GPU for the model's structure. 
    # Your interceptor will catch these initial cudaMalloc calls.
    print("Loading model to GPU...")
    model = models.resnet50().cuda()
    model.train() # Set to training mode so it calculates gradients (uses more memory)

    # We need a standard optimizer and loss function to simulate real training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # 3. The "Bursty" Training Loop
    # We loop multiple times to simulate a heavy, continuous workload
    for i in range(num_batches):
        print(f"  Processing batch {i+1}/{num_batches}...")

        # Create FAKE data (random numbers) shaped like standard images (3 colors, 224x224 pixels)
        # Moving these to ".cuda()" triggers more cudaMalloc calls.
        dummy_images = torch.randn(batch_size, 3, 224, 224).cuda()
        
        # Create FAKE labels (random answers for the AI to "guess")
        dummy_labels = torch.randint(0, 1000, (batch_size,)).cuda()

        # --- THE HEAVY MATH ---
        optimizer.zero_grad() 
        
        # Forward pass: push data through the model (Triggers compute kernels)
        outputs = model(dummy_images) 
        
        # Calculate how "wrong" the model was
        loss = criterion(outputs, dummy_labels)
        
        # Backward pass: calculate updates. This is highly memory intensive!
        loss.backward() 
        
        # Apply the updates
        optimizer.step()

        # --- MEMORY CLEANUP ---
        # We explicitly delete the fake data from this loop
        del dummy_images
        del dummy_labels
        del outputs
        del loss

        # IMPORTANT FOR YOUR PROJECT: 
        # PyTorch likes to hold onto memory secretly just in case it needs it again. 
        # empty_cache() forces PyTorch to actually return it to the GPU.
        # This will trigger the 'cudaFree' calls your C++ interceptor is looking for!
        torch.cuda.empty_cache()

    # 4. Stop the timer and calculate JCT
    end_time = time.time()
    jct = end_time - start_time

    print(f"--- Benchmark Complete ---")
    print(f"Total Job Completion Time (JCT): {jct:.2f} seconds")

if __name__ == "__main__":
    # You can increase num_batches to make the test run longer
    run_resnet_benchmark(num_batches=20, batch_size=32)