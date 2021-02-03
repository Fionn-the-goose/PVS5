// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono> //using this for sequential speed tests

#define DATA_SIZE   1000
#define MEM_SIZE    DATA_SIZE * DATA_SIZE * sizeof(float) 

const char* KernelSource =

"#define DIM 1000																		\n"
"__kernel void matmult(__global float* A, __global float* B, __global float* C,			\n"
"	__local float* Al)																	\n"
"{																						\n"
//"	float Bl[DIM];																		\n"
"	float sum;																			\n"
"	int i, j, k;																		\n"
"	j = get_global_id(0);																\n"
"	int il = get_local_id(0);															\n"
"	int nl = get_local_size(0);															\n"
"	for (i = 0; i < DIM; i++)															\n"
"	{																					\n"
"		for (k = il; k < DIM; k += nl)													\n"
"			Al[k] = A[i*DIM + k];														\n"
"		sum = 0.f;																		\n"
"		for (k = 0; k < DIM; k++)														\n"
//"			sum += A[i*DIM + k] * B[k*DIM + j];											\n"
"			sum += Al[k] * B[k*DIM + j];												\n"
"		C[i * DIM + j] = sum;															\n"
"	}																					\n"
"}																						\n"
"																						\n";

//void test(float* A, float* B, float* C, float* Al)
//{
//	const int DIM = 200;
//	float Al[DIM], sum;
//	int i, j, k;
//	j = 0;
//	for (i = 0; i < DIM; i++)
//	{
//		for (k = 0; k < DIM; k++)
//			Al[k] = A[i * DIM + k];
//		sum = 0.f;
//		for (k = 0; k < DIM; k++)
//			sum += Al[k] * B[k * DIM + j];
//		C[i * DIM + j] = sum;
//	}
//}


//"#define DATA_SIZE 200																	\n"
//"__kernel void matmult(__global float* Ap, __global float* Bp, __global float* Cp)		\n"
//"{																						\n"
//"	int i, j, k;																		\n"
//"	float sum = 0.f;																	\n"
//"	i = get_global_id(0);																\n"
//"	j = get_global_id(1);																\n"
//"	for (k = 0; k < DATA_SIZE; ++k)														\n"
//"	{																					\n"
//"		sum += Ap[i * DATA_SIZE + k] * Bp[k * DATA_SIZE + j];							\n"
//"	}																					\n"
//"	Cp[i * DATA_SIZE + j] = sum;														\n"
//"}																						\n"
//"																						\n";


float** alloc_mat(int row, int col)
{
	float** A1, * A2;

	A1 = (float**)calloc(row, sizeof(float*));
	A2 = (float*)calloc(row * col, sizeof(float));
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
			if (A[i][j] != B[i][j]) return false;
		}

	return true;
}

//hippity hoppity this ggt from the lecture is my property
int ggt(int x, int y)
{
	int z;
	while (y) {
		z = x % y;
		x = y;
		y = z;
	}
	return x;
}

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





	cl_int				err;
	cl_platform_id* platforms = NULL;
	char			    platform_name[1024];
	cl_device_id	    device_id = NULL;
	cl_uint			    num_of_platforms = 0, num_of_devices = 0;
	cl_context 			context;
	cl_kernel 			kernel;
	cl_command_queue	command_queue;
	cl_program 			program;
	size_t				global[1] = { DATA_SIZE }, local[1];
	float				results[DATA_SIZE] = { 0 };

	cl_event event;
	cl_ulong start, end;

	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

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

		for (unsigned int i = 0; i < num_of_platforms; i++)
		{
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}

			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}

		err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), local, NULL);
		if (err != CL_SUCCESS)
		{
			printf("I am sad. Error: %d\n", err);
			return 0;
		}
		local[0] = ggt(global[0], local[0]);
	}

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	kernel = clCreateKernel(program, "matmult", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}



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
	clSetKernelArg(kernel, 3, sizeof(cl_mem)*DATA_SIZE, NULL);



	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, local, 0, NULL, &event);

	clFinish(command_queue);

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
