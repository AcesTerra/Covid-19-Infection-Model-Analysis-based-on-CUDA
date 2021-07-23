#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <cmath>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

// Agent structure with all properties
struct agent {
	double infectionProb;			// [0.02, 0.03]
	double externalInfectionProb;	// [0.02, 0.03]
	double mortalityProb;			// [0.007, 0.07]
	double mobilityProb;			// [0.3, 0.5]
	double shortMobilityProb;		// [0.7, 0.9]
	int incubationTime;				// [5, 6]
	int recoveryTime;				// 14
	int infectionStatus;			// Non infected (0), infected (1), quarantine (-1), deseaced (-2), cured (2)
	double x;						// [0, p]
	double y;						// [0, q]
};

// Simulation parameters
#define numberOfAgents 1024
const int maxSimulationDays = 30;
const int maxMovementsPerDay = 10;
#define maximumRadiusForLocalMovements 5
const float infectionLimitDistance = 1;
#define p 500
#define q 500

// Function to generate random int numbers with CUDA
__device__ int generateRandomIntCUDA(int gID) {
	curandState_t state;
	curand_init((unsigned long long)clock() + gID, 0, 0, &state);
	int result = curand(&state);
	return abs(result);
}

// Function to generate random float numbers with CUDA
__device__ double generateRandomFloatCUDA(float min, float max)
{
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock() + gID, 0, 0, &state);

	double result = curand_uniform_double(&state) * (max - min) + min;
	return result;
}

// Function to check CUDA errors
__host__ void check_CUDA_error(const char* msj) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %d %s (%s)", error, cudaGetErrorString(error), msj);
	}
}

// Function to move a short distance an agent
__device__ float shortMovement(float pos) {
	float newPos = (2 * generateRandomFloatCUDA(0.0, 1.0) - 1) * maximumRadiusForLocalMovements + pos;
	if (newPos > 500)
		newPos = 500;
	if (newPos < 0)
		newPos = 0;
	return newPos;
}

// Function to move a long distance in X an agent
__device__ float longXMovement(float pos) {
	float newPos = p * generateRandomFloatCUDA(-1.0, 1.0) + pos;
	if (newPos > 500)
		newPos = 500;
	if (newPos < 0)
		newPos = 0;
	return newPos;
}

// Function to move a long distance in Y an agent
__device__ float longYMovement(float pos) {
	float newPos = q * generateRandomFloatCUDA(-1.0, 1.0) + pos;
	if (newPos > 500)
		newPos = 500;
	if (newPos < 0)
		newPos = 0;
	return newPos;
}

// Function to initalize agents properties
__global__ void initializeAgents(agent allAgents[]) {
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	allAgents[gID].infectionProb = generateRandomFloatCUDA(0.02, 0.03);
	allAgents[gID].externalInfectionProb = generateRandomFloatCUDA(0.02, 0.03);
	allAgents[gID].mortalityProb = generateRandomFloatCUDA(0.007, 0.07);
	allAgents[gID].mobilityProb = generateRandomFloatCUDA(0.3, 0.5);
	allAgents[gID].shortMobilityProb = generateRandomFloatCUDA(0.7, 0.9);
	allAgents[gID].incubationTime = generateRandomIntCUDA(gID) % 2 + 5;
	allAgents[gID].recoveryTime = 14;
	allAgents[gID].infectionStatus = 0;
	allAgents[gID].x = generateRandomFloatCUDA(0.0, (float)p);
	allAgents[gID].y = generateRandomFloatCUDA(0.0, (float)q);
}
// Function to show all agents properties
__host__ void showAgents(agent allAgents[]) {
	for (int i = 0; i < numberOfAgents; i++) {
		printf("Agent's no. %d probability of infection: %f\n", i + 1, allAgents[i].infectionProb);
		printf("Agent's no. %d external probability of infection: %f\n", i + 1, allAgents[i].externalInfectionProb);
		printf("Agent's no. %d probability of mortality: %f\n", i + 1, allAgents[i].mortalityProb);
		printf("Agent's no. %d probability of mobility: %f\n", i + 1, allAgents[i].mobilityProb);
		printf("Agent's no. %d probability of short mobility: %f\n", i + 1, allAgents[i].shortMobilityProb);
		printf("Agent's no. %d incubation time: %d\n", i + 1, allAgents[i].incubationTime);
		printf("Agent's no. %d recovery time: %d\n", i + 1, allAgents[i].recoveryTime);
		printf("Agent's no. %d infection status: %d\n", i + 1, allAgents[i].infectionStatus);
		printf("Agent's no. %d x position: %f\n", i + 1, allAgents[i].x);
		printf("Agent's no. %d y position: %f\n\n", i + 1, allAgents[i].y);
	}
}

// Rule 1: Infection
__global__ void ruleOne(agent agents[], int historyCounter[]) {
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int j = 0; j < numberOfAgents; j++) {
		double distance = sqrt(pow(agents[gID].x - agents[j].x, 2.0) + pow(agents[gID].y - agents[j].y, 2.0));
		if (distance <= 1.0 && agents[j].infectionStatus == 1 && agents[gID].infectionStatus == 0 && gID != j) {
			float infection = generateRandomFloatCUDA(0.0, 1.0);
			if (infection <= agents[gID].infectionProb) {
				agents[gID].infectionStatus = 1;
				historyCounter[gID] = 1;
			}
		}
	}
}
// Rule 2: Mobility
__global__ void ruleTwo(agent agents[]) {
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	float movProb = generateRandomFloatCUDA(0.0, 1.0);
	if (movProb <= agents[gID].mobilityProb && (agents[gID].infectionStatus == 0 || agents[gID].infectionStatus == 1)) {
		float shortMovProb = generateRandomFloatCUDA(0.0, 1.0);
		float newXPos, newYPos;
		if (shortMovProb <= agents[gID].shortMobilityProb) {
			newXPos = shortMovement(agents[gID].x);
			newYPos = shortMovement(agents[gID].y);
			agents[gID].x = newXPos;
			agents[gID].y = newYPos;
		}
		else {
			newXPos = longXMovement(agents[gID].x);
			newYPos = longYMovement(agents[gID].y);
			agents[gID].x = newXPos;
			agents[gID].y = newYPos;
		}
	}
}

// Rule 3: External infection
__global__ void ruleThree(agent agents[], int historyCounter[]) {
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	float infectionExternal = generateRandomFloatCUDA(0.0, 1.0);
	if (infectionExternal <= agents[gID].externalInfectionProb && agents[gID].infectionStatus == 0) {
		agents[gID].infectionStatus = 1;
		historyCounter[gID] = 1;
	}
}

// Rule 4: Incucation time, symptoms, quarantine and recovery time
__global__ void ruleFour(agent agents[], int historyCounter[]) {
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	if (agents[gID].infectionStatus == -1 && agents[gID].recoveryTime > 0) {
		agents[gID].recoveryTime = agents[gID].recoveryTime - 1;
	}
	if (agents[gID].infectionStatus == 1 && agents[gID].incubationTime > 0) {
		agents[gID].incubationTime = agents[gID].incubationTime - 1;
	}
	if (agents[gID].infectionStatus == 1 && agents[gID].incubationTime == 0) {
		agents[gID].infectionStatus = -1;
	}
	if (agents[gID].infectionStatus == -1 && agents[gID].recoveryTime == 0) {
		agents[gID].infectionStatus = 2;
		historyCounter[gID] = 1;
	}
}
// Rule 5: Fatal cases
__global__ void ruleFive(agent agents[], int historyCounter[]) {
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	float fatal = generateRandomFloatCUDA(0.0, 1.0);
	if (fatal <= agents[gID].mortalityProb && agents[gID].infectionStatus == -1) {
		agents[gID].infectionStatus = -2;
		historyCounter[gID] = 1;
	}
}

//Function to sum the history of specific day
__global__ void sumHistory(int historyCounter[]) {
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();
	int jump = numberOfAgents / 2;
	while (jump) {
		if (gID < jump) {
			historyCounter[gID] = historyCounter[gID] + historyCounter[gID + jump];
		}
		__syncthreads();
		jump = jump / 2;
	}
}

// Function to update the history of pandemic
__global__ void updateHistory(int day, int historyCounter[], int historyToUpdate[]) {
	historyToUpdate[day] = historyCounter[0];
}

// Function to initialize device counters
__global__ void initializeHistory(int counterControl_dev[], int historyCounter_dev[]) {
	int gID = blockIdx.x * blockDim.x + threadIdx.x;
	counterControl_dev[gID] = 0;
	historyCounter_dev[gID] = 0;
}

int main() {
	/*
	*************************************************************
	******************** Initalization phase ********************
	*************************************************************
	*/

	cudaEvent_t start_GPU;
	cudaEvent_t end_GPU;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	int allInfectionsCounter_host = 0;
	int* infectionHistory_host;
	infectionHistory_host = (int*)malloc(maxSimulationDays * sizeof(int));
	int allRecoveryCounter_host = 0;
	int* recoveryHistory_host;
	recoveryHistory_host = (int*)malloc(maxSimulationDays * sizeof(int));
	int allFatalCounter_host = 0;
	int* fatalHistory_host;
	fatalHistory_host = (int*)malloc(maxSimulationDays * sizeof(int));

	int* allInfectionsCounter_dev;
	cudaMalloc((void**)&allInfectionsCounter_dev, sizeof(int));
	int* infectionHistory_dev;
	cudaMalloc((void**)&infectionHistory_dev, maxSimulationDays * sizeof(int));
	int* allRecoveryCounter_dev;
	cudaMalloc((void**)&allRecoveryCounter_dev, sizeof(int));
	int* recoveryHistory_dev;
	cudaMalloc((void**)&recoveryHistory_dev, maxSimulationDays * sizeof(int));
	int* allFatalCounter_dev;
	cudaMalloc((void**)&allFatalCounter_dev, sizeof(int));
	int* fatalHistory_dev;
	cudaMalloc((void**)&fatalHistory_dev, maxSimulationDays * sizeof(int));

	agent* allAgents_dev;
	cudaMalloc((void**)&allAgents_dev, numberOfAgents * sizeof(agent));

	int* counterControl_dev, * historyCounter_dev;
	cudaMalloc((void**)&counterControl_dev, numberOfAgents * sizeof(int));
	cudaMalloc((void**)&historyCounter_dev, numberOfAgents * sizeof(int));

	dim3 block(32);
	dim3 grid(32);

	initializeAgents << <grid, block >> > (allAgents_dev);
	initializeHistory << <grid, block >> > (counterControl_dev, historyCounter_dev);
	cudaDeviceSynchronize();
	check_CUDA_error("Error en kernel");

	printf("---------------------Simulation parameters---------------------\n");
	printf("\nNumber of agents: %d\n", numberOfAgents);
	printf("Simulation days: %d\n", maxSimulationDays);
	printf("Max movements per day: %d\n", maxMovementsPerDay);
	printf("Maximum radius for local movements: %d\n", maximumRadiusForLocalMovements);
	printf("Infection limit distance: %f\n", infectionLimitDistance);
	printf("P: %d\n", p);
	printf("Q: %d\n", q);
	printf("\n--------------------Initializing simulation--------------------\n");

	/*
	*************************************************************
	********************** Operation phase **********************
	*************************************************************
	*/

	for (int day = 0; day < maxSimulationDays; day++) {
		for (int mov = 0; mov < maxMovementsPerDay; mov++) {
			ruleOne << <grid, block >> > (allAgents_dev, historyCounter_dev);
			cudaDeviceSynchronize();
			check_CUDA_error("Error en kernel");
			ruleTwo << <grid, block >> > (allAgents_dev);
			cudaDeviceSynchronize();
			check_CUDA_error("Error en kernel");
		}
		ruleThree << <grid, block >> > (allAgents_dev, historyCounter_dev);
		sumHistory << <grid, block >> > (historyCounter_dev);
		updateHistory << <1, 1 >> > (day, historyCounter_dev, infectionHistory_dev);
		cudaDeviceSynchronize();
		cudaMemcpy(historyCounter_dev, counterControl_dev, numberOfAgents * sizeof(int), cudaMemcpyDeviceToDevice);
		check_CUDA_error("Error en kernel");
		ruleFour << <grid, block >> > (allAgents_dev, historyCounter_dev);
		sumHistory << <grid, block >> > (historyCounter_dev);
		updateHistory << <1, 1 >> > (day, historyCounter_dev, recoveryHistory_dev);
		cudaDeviceSynchronize();
		cudaMemcpy(historyCounter_dev, counterControl_dev, numberOfAgents * sizeof(int), cudaMemcpyDeviceToDevice);
		check_CUDA_error("Error en kernel");
		ruleFive << <grid, block >> > (allAgents_dev, historyCounter_dev);
		sumHistory << <grid, block >> > (historyCounter_dev);
		updateHistory << <1, 1 >> > (day, historyCounter_dev, fatalHistory_dev);
		cudaDeviceSynchronize();
		cudaMemcpy(historyCounter_dev, counterControl_dev, numberOfAgents * sizeof(int), cudaMemcpyDeviceToDevice);
		check_CUDA_error("Error en kernel");
	}

	/*
	*************************************************************
	************************ Show results ***********************
	*************************************************************
	*/
	printf("\n---------------------Simulation terminated---------------------\n");

	cudaMemcpy(infectionHistory_host, infectionHistory_dev, maxSimulationDays * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(recoveryHistory_host, recoveryHistory_dev, maxSimulationDays * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(fatalHistory_host, fatalHistory_dev, maxSimulationDays * sizeof(int), cudaMemcpyDeviceToHost);
	check_CUDA_error("Error cudaMemcpy");

	for (int i = 0; i < maxSimulationDays; i++) {
		allInfectionsCounter_host += infectionHistory_host[i];
		allRecoveryCounter_host += recoveryHistory_host[i];
		allFatalCounter_host += fatalHistory_host[i];
	}

	int zeroDayInfected = 0, halfPopulationInfected = 0, allPopulationInfected = 0;
	int zeroDayRecovered = 0, halfAgentsRecovered = 0, allAgentsRecovered = 0;
	int zeroDayFatal = 0, halfAgentsFatal = 0, allAgentsFatal = 0;
	int halfPopulationInfectedDay = 0, allPopulationInfectedDay = 0;
	int halfAgentsRecoveredDay = 0, allAgentsRecoveredDay = 0;
	int halfAgentsFatalDay = 0, allAgentsFatalDay = 0;
	bool zeroDayInfectedFlag = false, halfPopulationInfectedFlag = false;
	bool zeroDayRecoveredFlag = false, halfAgentsRecoveredFlag = false;
	bool zeroDayFatalFlag = false, halfAgentsFatalFlag = false;

	printf("\nTotal infected cases: %d\n", allInfectionsCounter_host);
	printf("Infection history: ");
	for (int i = 0; i < maxSimulationDays; i++) {
		printf("%d ", infectionHistory_host[i]);
		halfPopulationInfected += infectionHistory_host[i];
		allPopulationInfected += infectionHistory_host[i];
		if (infectionHistory_host[i] > 0 && !zeroDayInfectedFlag) {
			zeroDayInfected = i + 1;
			zeroDayInfectedFlag = true;
		}
		if (halfPopulationInfected >= (numberOfAgents / 2) && !halfPopulationInfectedFlag) {
			halfPopulationInfectedDay = i + 1;
			halfPopulationInfectedFlag = true;
		}
		if (allPopulationInfected == numberOfAgents)
			allPopulationInfectedDay = i + 1;
	}
	printf("\nZero day infection case: %d\n", zeroDayInfected);
	printf("Half population infected day: %d\n", halfPopulationInfectedDay);
	printf("All population infected day: %d\n", allPopulationInfectedDay);

	printf("\nTotal recovery cases: %d\n", allRecoveryCounter_host);
	printf("Recovery history: ");
	for (int i = 0; i < maxSimulationDays; i++) {
		printf("%d ", recoveryHistory_host[i]);
		halfAgentsRecovered += recoveryHistory_host[i];
		allAgentsRecovered += recoveryHistory_host[i];
		if (recoveryHistory_host[i] > 0 && !zeroDayRecoveredFlag) {
			zeroDayRecovered = i + 1;
			zeroDayRecoveredFlag = true;
		}
		if (halfAgentsRecovered >= (allRecoveryCounter_host / 2) && !halfAgentsRecoveredFlag) {
			halfAgentsRecoveredDay = i + 1;
			halfAgentsRecoveredFlag = true;
		}
		if (allAgentsRecovered == allRecoveryCounter_host)
			allAgentsRecoveredDay = i + 1;
	}
	printf("\nZero day recovery case: %d\n", zeroDayRecovered);
	printf("Half agents recovered day: %d\n", halfAgentsRecoveredDay);
	printf("All agents recovered day: %d\n", allAgentsRecoveredDay);

	printf("\nTotal fatal cases: %d\n", allFatalCounter_host);
	printf("Fatal history: ");
	for (int i = 0; i < maxSimulationDays; i++) {
		printf("%d ", fatalHistory_host[i]);
		halfAgentsFatal += fatalHistory_host[i];
		allAgentsFatal += fatalHistory_host[i];
		if (fatalHistory_host[i] > 0 && !zeroDayFatalFlag) {
			zeroDayFatal = i + 1;
			zeroDayFatalFlag = true;
		}
		if (halfAgentsFatal >= (allFatalCounter_host / 2) && !halfAgentsFatalFlag) {
			halfAgentsFatalDay = i + 1;
			halfAgentsFatalFlag = true;
		}
		if (allAgentsFatal == allFatalCounter_host)
			allAgentsFatalDay = i + 1;
	}
	printf("\nZero day fatal case: %d\n", zeroDayFatal);
	printf("Half agents fatal day: %d\n", halfAgentsFatalDay);
	printf("All agents fatal day: %d\n", allAgentsFatalDay);

	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	float elapsedTime_GPU;
	cudaEventElapsedTime(&elapsedTime_GPU, start_GPU, end_GPU);
	printf("\nTime GPU: %f miliseconds. \n", elapsedTime_GPU);

	cudaEventDestroy(start_GPU);
	cudaEventDestroy(end_GPU);

	free(infectionHistory_host);
	free(recoveryHistory_host);
	free(fatalHistory_host);

	cudaFree(allAgents_dev);
	cudaFree(allInfectionsCounter_dev);
	cudaFree(infectionHistory_dev);
	cudaFree(allRecoveryCounter_dev);
	cudaFree(recoveryHistory_dev);
	cudaFree(allFatalCounter_dev);
	cudaFree(fatalHistory_dev);
	cudaFree(counterControl_dev);
	cudaFree(historyCounter_dev);

	return 0;
}