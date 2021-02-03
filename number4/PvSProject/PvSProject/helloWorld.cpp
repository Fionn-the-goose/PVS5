// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              //includes open CL library to enable open CL functionality
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono> //using this for sequential version speed test

#define DATA_SIZE   1000                         //prepare matrix size
#define MEM_SIZE    DATA_SIZE * DATA_SIZE * sizeof(float)   //prepares memory needed to store a matrix of given size containing float values

/** kernel's source text as string **/
const char* KernelSource =

"#define DATA_SIZE 1000																	\n"
"__kernel void matmult(__global float* Ap, __global float* Bp, __global float* Cp)		\n"
"{																						\n"
"	int i, j, k;																		\n"
"	float sum = 0.f;																	\n"
"	i = get_global_id(0);																\n" //past a certain threshhold of matrix size, doubling the matrix size causes an eightfold increase in compile time, suggesting that we eventually reach the maximum possible parallelization and return to a structure equivalent to 3 nested loops, up until that point it is significantly faster though (for us this happened when going from size 1000 to 2000)
"	j = get_global_id(1);																\n"
"	for (k = 0; k < DATA_SIZE; ++k)														\n"
"	{																					\n"
"		sum += Ap[i * DATA_SIZE + k] * Bp[k * DATA_SIZE + j];							\n"
"	}																					\n"
"	Cp[i * DATA_SIZE + j] = sum;														\n"
"}																						\n"
"																						\n";

//"#define DATA_SIZE 3												\n"
//"__kernel void test(__global float *input, __global float *input2, __global float *output)  \n"
//"{																	\n"
//"	size_t i = get_global_id(0);									\n"
//"	output[i] = input[i] * input2[i];								\n"
//"}																	\n"
//"\n";

float** alloc_mat(int row, int col)
{
	float** A1, * A2;

	A1 = (float**)calloc(row, sizeof(float*));	 // pointer on rows
	A2 = (float*)calloc(row * col, sizeof(float));    // all matrix elements
	for (int i = 0; i < row; i++)
		A1[i] = A2 + i * col;

	return A1;
}

void init_mat(float** A, int row, int col)
{
	for (int i = 0; i < row * col; i++)
		A[0][i] = (float)(rand() % 10);
}

void init_zero(float** A, int row, int col)
{
	for (int i = 0; i < row * col; i++)
		A[0][i] = 0;
}

void print_mat(float** A, int row, int col, char const* tag)
{
	int i, j;

	printf("Matrix %s:\n", tag);
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
			printf("%6.1f   ", A[i][j]);
		printf("\n");
	}
}

void free_mat(float** A, int num_rows) {
	free(A[0]);
	free(A);
}

bool compare_mat(float** A, float** B, int row, int col) {
	for(int i = 0; i < row; ++i)
		for (int j = 0; j < col; ++j)
		{
			if (A[i][j] != B[i][j]) return false; //iterate through all elements, if any are different return that they are not equal
		}

	return true; //if we reached this point we haven't found any differences as we would have otherwise returned false already, so we can return true
}


/** Body of the main code **/
int main(void)
{
	//prepare matrices
	float** A = alloc_mat(DATA_SIZE, DATA_SIZE); init_mat(A, DATA_SIZE, DATA_SIZE);
	float** B = alloc_mat(DATA_SIZE, DATA_SIZE); init_mat(B, DATA_SIZE, DATA_SIZE);
	float** serialC = alloc_mat(DATA_SIZE, DATA_SIZE);
	//Serial variant in here
	{
		std::chrono::milliseconds start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		int i, j, k;
		for (i = 0; i < DATA_SIZE; i++)
			for (j = 0; j < DATA_SIZE; j++)
				for (k = 0; k < DATA_SIZE; k++)
					serialC[i][j] += A[i][k] * B[k][j];

		std::chrono::milliseconds end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("\nSerial Time Taken in Milliseconds: %lld\n\n\n", end.count() - start.count());
	}
	//Everything past here is for the open cl version





	cl_int				err;                      //stores information about success or failure of commands
	cl_platform_id* platforms = NULL;         //id of the platform
	char			    platform_name[1024];      //name of the platform
	cl_device_id	    device_id = NULL;         //id of the device
	cl_uint			    num_of_platforms = 0,     //number of platforms
		num_of_devices = 0;       //number of devices
	cl_context 			context;                  //context
	cl_kernel 			kernel;                   //kernel
	cl_command_queue	command_queue;            //queue storing commands
	cl_program 			program;                  //stores generated program
	//cl_mem				input, input2, output;            //stores necessary available memory
	//float				data[DATA_SIZE] =         //prepare storage for data
	//{ 1, 2, 3 , 4, 5, 6, 7, 8, 9, 10};
	size_t				global[2] = { DATA_SIZE, DATA_SIZE };  //global storage for data size
	float				results[DATA_SIZE] = { 0 }; //empty storage with enough space for future matrix data

	cl_event event;
	cl_ulong start, end;

	/* 1) */

	// gets the number of available platforms, returning if it was successful
	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	// gets the id of the current platform
	platforms = (cl_platform_id*)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}
	else
	{
		int nvidia_platform = 0;

		// For every platform
		for (unsigned int i = 0; i < num_of_platforms; i++)
		{
			// Attempt to get its information
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}

			// check if using nvidia
			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

		// Get ID of current device, and maximum available numbers
		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	// Open Context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	// Creates a queue (FIFO)
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	// Generate online program
	program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

	// Compile and link the kernel source text
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	// Specify kernel
	kernel = clCreateKernel(program, "matmult", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}


	/* 2) */

	//float** A, ** B;
	float** C;
	cl_mem Ap, Bp, Cp;

	//A = alloc_mat(DATA_SIZE, DATA_SIZE); init_mat(A, DATA_SIZE, DATA_SIZE);
	//B = alloc_mat(DATA_SIZE, DATA_SIZE); init_mat(B, DATA_SIZE, DATA_SIZE);
	C = alloc_mat(DATA_SIZE, DATA_SIZE);

	Ap = clCreateBuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
	Bp = clCreateBuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
	Cp = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE, NULL, &err);

	clEnqueueWriteBuffer(command_queue, Ap, CL_TRUE, 0, MEM_SIZE, A[0], 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, Bp, CL_TRUE, 0, MEM_SIZE, B[0], 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &Ap);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &Bp);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &Cp);

	// Create Buffer for input and output
	//input = clCreateBuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
	//input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
	//output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, &err);

	//// Write related data from 'data' into 'input' buffer
	//clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, MEM_SIZE, data, 0, NULL, NULL);
	//clEnqueueWriteBuffer(command_queue, input2, CL_TRUE, 0, MEM_SIZE, data, 0, NULL, NULL);

	//// Define the order of operations of kernel arguments
	//clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	//clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	//clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);


	/* 3)  */

	// Puts kernel into command queue and splits up instructions
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);

	// Wait for queue to complete
	clFinish(command_queue);

	// Read and store results of output buffer into results
	//clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, MEM_SIZE, results, 0, NULL, NULL);
	clEnqueueReadBuffer(command_queue, Cp, CL_TRUE, 0, MEM_SIZE, C[0], 0, NULL, NULL);
	// Display results
	//for (unsigned int i = 0; i < DATA_SIZE; i++)
	//	printf("%f\n", results[i]);

	//print_mat(A, DATA_SIZE, DATA_SIZE, "A");
	//print_mat(B, DATA_SIZE, DATA_SIZE, "B");
	//print_mat(C, DATA_SIZE, DATA_SIZE, "C");
	//print_mat(serialC, DATA_SIZE, DATA_SIZE, "sC");

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	printf("OpenCL time = %.1f ms\n", ((end - start) / 1000000.0));

	printf("Matrices are %s", compare_mat(C, serialC, DATA_SIZE, DATA_SIZE) ? "equal" : "not equal");

	/* 4) */
	clReleaseMemObject(Ap);
	clReleaseMemObject(Bp);
	clReleaseMemObject(Cp);
	//clReleaseMemObject(input);
	//clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	free_mat(A, DATA_SIZE);
	free_mat(B, DATA_SIZE);
	free_mat(C, DATA_SIZE);
	free_mat(serialC, DATA_SIZE);

	return 0;
}
