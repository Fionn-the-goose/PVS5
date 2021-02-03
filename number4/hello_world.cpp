// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              //
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_SIZE   10                          //
#define MEM_SIZE    DATA_SIZE * sizeof(float)   //

/** **/ 
const char *KernelSource =
	"#define DATA_SIZE 10												\n"
	"__kernel void test(__global float *input, __global float *output)  \n"
	"{																	\n"
	"	size_t i = get_global_id(0);									\n"
	"	output[i] = input[i] * input[i];								\n"
	"}																	\n"
	"\n";

/** **/
int main (void)
{
	cl_int				err;                      //
	cl_platform_id*		platforms = NULL;         //
	char			    platform_name[1024];      //
	cl_device_id	    device_id = NULL;         //
	cl_uint			    num_of_platforms = 0,     //
					    num_of_devices = 0;       //
	cl_context 			context;                  //
	cl_kernel 			kernel;                   //
	cl_command_queue	command_queue;            //
	cl_program 			program;                  //
	cl_mem				input, output;            //
	float				data[DATA_SIZE] =         //
							{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	size_t				global[1] = {DATA_SIZE};  //
	float				results[DATA_SIZE] = {0}; //

	/* 1) */

	// Abfragen der Anzahl verfügbarer Plattformen und ihrer ID
	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	// 
	platforms = (cl_platform_id *)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}
	else
	{
		int nvidia_platform = 0;

		// 
		for (unsigned int i=0; i<num_of_platforms; i++)
		{
			//
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name,	NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}
			
			// 
			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

		// Abfragen der Anzahl verfügbarer Geräte und ihrer ID
		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	// ... Kontext oeffnen
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	// Erzeugen einer Befehlswarteschlange (FIFO) 
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	// ... Erzeuge online ein Programm vom Quellcode des Kerns 
	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

  //Kompiliere und Linke den Kernel-Quelltext
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	//Definiere den Kernel Einsprungspunkt
	kernel = clCreateKernel(program, "test", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}


	/* 2) */

	// ... Erzeuge Puffer für Ein-und Ausgabe 
	input  = clCreateBuffer (context, CL_MEM_READ_ONLY,	 MEM_SIZE, NULL, &err);
	output = clCreateBuffer (context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, &err);

	// Kopiere zusammenhängende Daten aus 'data' in den Eingabe-Puffer 'input' 
	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, MEM_SIZE, data, 0, NULL, NULL);

	//  Definiere die Reihenfolge der Argumente des Kerns: hello(input, output)
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);


	/* 3)  */

	// ... Einreihen des Kerns in die Befehlswarteschlange und Aufteilungsbereich angeben 
	clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

	// Auf die Beendigung der Operation warten 
	clFinish(command_queue);

	// Kopiere die Ergebnisse vom Ausgabe-Puffer 'output' in das Ergebnisfeld 'results' 
	clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, MEM_SIZE, results, 0, NULL, NULL);

  //... Aufraeumender    OpenCLRessourcen
  for (unsigned int i=0; i < DATA_SIZE; i++)
    printf("%f\n", results[i]);


	/* 4) */
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
