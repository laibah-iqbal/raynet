#include <GymApi.h>
#include <omnetpp.h>
#include <unordered_map>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <sstream>

using namespace std;
using namespace omnetpp;

// Function to generate a random integer within a given range
int generateRandomInt(int min, int max) {
    return rand() % (max - min + 1) + min;
}

// Function to generate a random double within a given range
double generateRandomDouble(double min, double max) {
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

// Function to replace specific text in the INI file with random values
void generateRandomConfig(const std::string& filePath) {
    // Generate random values for the parameters
    int minDelay = generateRandomInt(1, 100); // ms
    int linkRate = generateRandomInt(1, 1000); // Mbps
    int queuePacketCapacity = generateRandomInt(1, 1000); // packets

    // Read the content of the file
    std::ifstream inputFile(filePath);
    std::stringstream buffer;
    buffer << inputFile.rdbuf();
    std::string content = buffer.str();
    inputFile.close();

    // Replace the specific text with the random values
    size_t pos;
    if ((pos = content.find("DELAY_PLACEHOLDER")) != std::string::npos) {
        size_t end = content.find("\n", pos);
        content.replace(pos, end - pos, std::to_string(minDelay) + " ms");
    }
    if ((pos = content.find("LINKRATE_PLACEHOLDER")) != std::string::npos) {
        size_t end = content.find("\n", pos);
        content.replace(pos, end - pos, std::to_string(linkRate) + " Mbps");
    }
    if ((pos = content.find("Q_PLACEHOLDER")) != std::string::npos) {
        size_t end = content.find("\n", pos);
        content.replace(pos, end - pos, std::to_string(queuePacketCapacity));
    }

    // Write the new content back to the file
    std::ofstream outputFile(filePath);
    outputFile << content;
    outputFile.close();
}

int main(int argc, char **argv) {
    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    std::string _iniPath;
    ObsType obs;

    _iniPath = (string(getenv("HOME")) + string("/raynet/configs/orca/orcaConfigStatic.ini")).c_str();
    std::cout << _iniPath << std::endl;

    // Generate random values and update the INI file
    generateRandomConfig(_iniPath);

    GymApi* gymapi = new GymApi();
    gymapi->initialise(_iniPath);
    auto id_obs = gymapi->reset();

    std::vector<std::string> keys;
    keys.reserve(id_obs.size());

    std::vector<ObsType> vals;
    vals.reserve(id_obs.size());

    for (auto kv : id_obs) {
        keys.push_back(kv.first);
        vals.push_back(kv.second);
    }

    std::string agentId = keys.front();

    bool done = false;
    bool simDone = false;
    while (!done && strcmp(agentId.c_str(), "SIMULATION_END") != 0 && !simDone) {
        for (std::unordered_map<std::string, ObsType>::iterator it = id_obs.begin(); it != id_obs.end(); ++it) {
            agentId = it->first;
        }
        std::unordered_map<std::string, ActionType> actions({{agentId, 1}});
        auto ret = gymapi->step(actions);
        done = std::get<2>(ret)["__all__"];
        obs = std::get<0>(ret)[agentId];
        simDone = std::get<3>(ret)["simDone"];
    }

    gymapi->shutdown();
    gymapi->cleanupmemory();

    return 0;
}
