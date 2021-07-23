
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctime>
//#include <math.h>
#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

// Agent structure with all properties
struct agent {
	float infectionProb;			// [0.02, 0.03]
	float externalInfectionProb;	// [0.02, 0.03]
	float mortalityProb;			// [0.007, 0.07]
	float mobilityProb;				// [0.3, 0.5]
	float shortMobilityProb;		// [0.7, 0.9]
	int incubationTime;				// [5, 6]
	int recoveryTime;				// 14
	int infectionStatus;			// Non infected (0), infected (1), quarantine (-1), deseaced (-2), cured (2)
	float x;						// [0, p]
	float y;						// [0, q]
};

// Simulation parameters
const int numberOfAgents = 1024;
const int maxSimulationDays = 30;
const int maxMovementsPerDay = 10;
const float maximumRadiusForLocalMovements = 5;
const float infectionLimitDistance = 1;
const float p = 500;
const float q = 500;

int allInfectionsCounter = 0;
int infectionsPerDay = 0;
int infectionHistory[maxSimulationDays];
int allRecoveryCounter = 0;
int recoveryPerDay = 0;
int recoveryHistory[maxSimulationDays];
int allFatalCounter = 0;
int fatalPerDay = 0;
int fatalHistory[maxSimulationDays];

// Function to generate a random float between a range
float generateRandom(float a, float b) {
	float r = a + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (b - a)));
	return r;
}

// Function to move a short distance an agent
float shortMovement(float pos) {
	float newPos = (2 * generateRandom(0.0, 1.0) - 1) * maximumRadiusForLocalMovements + pos;
	if (newPos > 500)
		newPos = 500;
	if (newPos < 0)
		newPos = 0;
	return newPos;
}

// Function to move a long distance in X an agent
float longXMovement(float pos) {
	float newPos = p * generateRandom(-1.0, 1.0) + pos;
	if (newPos > 500)
		newPos = 500;
	if (newPos < 0)
		newPos = 0;
	return newPos;
}

// Function to move a long distance in Y an agent
float longYMovement(float pos) {
	float newPos = q * generateRandom(-1.0, 1.0) + pos;
	if (newPos > 500)
		newPos = 500;
	if (newPos < 0)
		newPos = 0;
	return newPos;
}

// Test function to initalize infected agents
void initializeInfectedAgents(agent allAgents[]) {
	for (int i = 0; i < numberOfAgents; i++) {
		allAgents[i].infectionProb = generateRandom(0.02, 0.03);
		allAgents[i].externalInfectionProb = generateRandom(0.02, 0.03);
		allAgents[i].mortalityProb = generateRandom(0.007, 0.07);
		allAgents[i].mobilityProb = generateRandom(0.3, 0.5);
		allAgents[i].shortMobilityProb = generateRandom(0.7, 0.9);
		allAgents[i].incubationTime = rand() % 2 + 5;
		allAgents[i].recoveryTime = 14;
		allAgents[i].infectionStatus = rand() % 2;
		allAgents[i].x = rand() % (int)p + 1;
		allAgents[i].y = rand() % (int)q + 1;
	}
}

// Function to initialize all agent properties
void initializeAgents(agent allAgents[]) {
	for (int i = 0; i < numberOfAgents; i++) {
		allAgents[i].infectionProb = generateRandom(0.02, 0.03);
		allAgents[i].externalInfectionProb = generateRandom(0.02, 0.03);
		allAgents[i].mortalityProb = generateRandom(0.007, 0.07);
		allAgents[i].mobilityProb = generateRandom(0.3, 0.5);
		allAgents[i].shortMobilityProb = generateRandom(0.7, 0.9);
		allAgents[i].incubationTime = rand() % 2 + 5;
		allAgents[i].recoveryTime = 14;
		allAgents[i].infectionStatus = 0;
		allAgents[i].x = rand() % (int)p + 1;
		allAgents[i].y = rand() % (int)q + 1;
	}
}

// Function to show all agents properties
void showAgents(agent allAgents[]) {
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
void ruleOne(agent agents[]) {
	for (int i = 0; i < numberOfAgents; i++) {
		for (int j = 0; j < numberOfAgents; j++) {
			double distance = sqrt(pow(agents[i].x - agents[j].x, 2.0) + pow(agents[i].y - agents[j].y, 2.0));
			if (distance <= 1.0 && agents[j].infectionStatus == 1 && agents[i].infectionStatus == 0 && i != j) {
				float infection = generateRandom(0.0, 1.0);
				if (infection <= agents[i].infectionProb) {
					agents[i].infectionStatus = 1;
					allInfectionsCounter++;
					infectionsPerDay++;
				}
			}
		}
	}
}

// Rule 2: Mobility
void ruleTwo(agent agents[]) {
	for (int i = 0; i < numberOfAgents; i++) {
		float movProb = generateRandom(0.0, 1.0);
		if (movProb <= agents[i].mobilityProb && (agents[i].infectionStatus == 0 || agents[i].infectionStatus == 1)) {
			float shortMovProb = generateRandom(0.0, 1.0);
			float newXPos, newYPos;
			if (shortMovProb <= agents[i].shortMobilityProb) {
				newXPos = shortMovement(agents[i].x);
				newYPos = shortMovement(agents[i].y);
				agents[i].x = newXPos;
				agents[i].y = newYPos;
			}
			else {
				newXPos = longXMovement(agents[i].x);
				newYPos = longYMovement(agents[i].y);
				agents[i].x = newXPos;
				agents[i].y = newYPos;
			}
		}
	}
}

// Rule 3: External infection
void ruleThree(agent agents[]) {
	for (int i = 0; i < numberOfAgents; i++) {
		float infectionExternal = generateRandom(0.0, 1.0);
		if (infectionExternal <= agents[i].externalInfectionProb && agents[i].infectionStatus == 0) {
			agents[i].infectionStatus = 1;
			allInfectionsCounter++;
			infectionsPerDay++;
		}
	}
}

// Rule 4: Incucation time, symptoms, quarantine and recovery time
void ruleFour(agent agents[]) {
	for (int i = 0; i < numberOfAgents; i++) {
		if (agents[i].infectionStatus == -1 && agents[i].recoveryTime > 0) {
			agents[i].recoveryTime = agents[i].recoveryTime - 1;
		}
		if (agents[i].infectionStatus == 1 && agents[i].incubationTime > 0) {
			agents[i].incubationTime = agents[i].incubationTime - 1;
		}
		if (agents[i].infectionStatus == 1 && agents[i].incubationTime == 0) {
			agents[i].infectionStatus = -1;
		}
		if (agents[i].infectionStatus == -1 && agents[i].recoveryTime == 0) {
			agents[i].infectionStatus = 2;
			allRecoveryCounter++;
			recoveryPerDay++;
		}
	}
}

// Rule 5: Fatal cases
void ruleFive(agent agents[]) {
	for (int i = 0; i < numberOfAgents; i++) {
		float fatal = generateRandom(0.0, 1.0);
		if (fatal <= agents[i].mortalityProb && agents[i].infectionStatus == -1) {
			agents[i].infectionStatus = -2;
			allFatalCounter++;
			fatalPerDay++;
		}
	}
}

int main() {
	/*
	*************************************************************
	******************** Initalization phase ********************
	*************************************************************
	*/
	clock_t start_CPU = clock();
	srand((int)time(0));
	agent* allAgents;
	allAgents = (agent*)malloc(numberOfAgents * sizeof(agent));
	initializeAgents(allAgents);

	printf("---------------------Simulation parameters---------------------\n");
	printf("\nNumber of agents: %d\n", numberOfAgents);
	printf("Simulation days: %d\n", maxSimulationDays);
	printf("Max movements per day: %d\n", maxMovementsPerDay);
	printf("Maximum radius for local movements: %f\n", maximumRadiusForLocalMovements);
	printf("Infection limit distance: %f\n", infectionLimitDistance);
	printf("P: %f\n", p);
	printf("Q: %f\n", q);
	printf("\n--------------------Initializing simulation--------------------\n");

	/*
	*************************************************************
	********************** Operation phase **********************
	*************************************************************
	*/

	for (int day = 0; day < maxSimulationDays; day++) {
		for (int mov = 0; mov < maxMovementsPerDay; mov++) {
			ruleOne(allAgents);
			ruleTwo(allAgents);
		}
		ruleThree(allAgents);
		ruleFour(allAgents);
		ruleFive(allAgents);
		infectionHistory[day] = infectionsPerDay;
		recoveryHistory[day] = recoveryPerDay;
		fatalHistory[day] = fatalPerDay;
		infectionsPerDay = 0;
		recoveryPerDay = 0;
		fatalPerDay = 0;
	}

	/*
	*************************************************************
	************************ Show results ***********************
	*************************************************************
	*/
	printf("\n---------------------Simulation terminated---------------------\n");

	int zeroDayInfected = 0, halfPopulationInfected = 0, allPopulationInfected = 0;
	int zeroDayRecovered = 0, halfAgentsRecovered = 0, allAgentsRecovered = 0;
	int zeroDayFatal = 0, halfAgentsFatal = 0, allAgentsFatal = 0;
	int halfPopulationInfectedDay = 0, allPopulationInfectedDay = 0;
	int halfAgentsRecoveredDay = 0, allAgentsRecoveredDay = 0;
	int halfAgentsFatalDay = 0, allAgentsFatalDay = 0;
	bool zeroDayInfectedFlag = false, halfPopulationInfectedFlag = false;
	bool zeroDayRecoveredFlag = false, halfAgentsRecoveredFlag = false;
	bool zeroDayFatalFlag = false, halfAgentsFatalFlag = false;

	printf("\nTotal infected cases: %d\n", allInfectionsCounter);
	printf("Infection history: ");
	for (int i = 0; i < maxSimulationDays; i++) {
		printf("%d ", infectionHistory[i]);
		halfPopulationInfected += infectionHistory[i];
		allPopulationInfected += infectionHistory[i];
		if (infectionHistory[i] > 0 && !zeroDayInfectedFlag) {
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

	printf("\nTotal recovery cases: %d\n", allRecoveryCounter);
	printf("Recovery history: ");
	for (int i = 0; i < maxSimulationDays; i++) {
		printf("%d ", recoveryHistory[i]);
		halfAgentsRecovered += recoveryHistory[i];
		allAgentsRecovered += recoveryHistory[i];
		if (recoveryHistory[i] > 0 && !zeroDayRecoveredFlag) {
			zeroDayRecovered = i + 1;
			zeroDayRecoveredFlag = true;
		}
		if (halfAgentsRecovered >= (allRecoveryCounter / 2) && !halfAgentsRecoveredFlag) {
			halfAgentsRecoveredDay = i + 1;
			halfAgentsRecoveredFlag = true;
		}
		if (allAgentsRecovered == allRecoveryCounter)
			allAgentsRecoveredDay = i + 1;
	}
	printf("\nZero day recovery case: %d\n", zeroDayRecovered);
	printf("Half agents recovered day: %d\n", halfAgentsRecoveredDay);
	printf("All agents recovered day: %d\n", allAgentsRecoveredDay);

	printf("\nTotal fatal cases: %d\n", allFatalCounter);
	printf("Fatal history: ");
	for (int i = 0; i < maxSimulationDays; i++) {
		printf("%d ", fatalHistory[i]);
		halfAgentsFatal += fatalHistory[i];
		allAgentsFatal += fatalHistory[i];
		if (fatalHistory[i] > 0 && !zeroDayFatalFlag) {
			zeroDayFatal = i + 1;
			zeroDayFatalFlag = true;
		}
		if (halfAgentsFatal >= (allFatalCounter / 2) && !halfAgentsFatalFlag) {
			halfAgentsFatalDay = i + 1;
			halfAgentsFatalFlag = true;
		}
		if (allAgentsFatal == allFatalCounter)
			allAgentsFatalDay = i + 1;
	}
	printf("\nZero day fatal case: %d\n", zeroDayFatal);
	printf("Half agents fatal day: %d\n", halfAgentsFatalDay);
	printf("All agents fatal day: %d\n", allAgentsFatalDay);


	clock_t end_CPU = clock();
	float elapsedTime_CPU = end_CPU - start_CPU;
	printf("\nTime CPU: %f miliseconds. \n", elapsedTime_CPU);
	free(allAgents);
	return 0;
}