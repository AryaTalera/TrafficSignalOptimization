# Traffic Signal Optimization with Reinforcement Learning

This project focuses on optimizing traffic signal timings using Reinforcement Learning (RL) to dynamically adapt to varying traffic conditions. The project leverages the SUMO (Simulation of Urban Mobility) simulator for realistic traffic simulation and trains a Deep Q-Network (DQN) agent to reduce congestion and waiting times.

---

## **Features**

- Dynamic traffic signal control using RL.
- Simulated real-world road networks with SUMO.
- Customized reward function for optimal traffic flow.
- Traffic route generation and dynamic flows for realistic scenarios.
- Comparative performance analysis between RL-based agent and baseline traffic signals.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/AryaTalera/TrafficSignalOptimization.git
cd TrafficSignalOptimization
```

### **2. Set Up Virtual Environment**
Create and activate a virtual environment (Python 3.10 or later is recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### **3. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **4. Install SUMO**
Download and install SUMO from the [official SUMO website](https://sumo.dlr.de/docs/Downloads.php). Ensure that SUMO and its tools (e.g., `randomTrips.py`) are added to your system PATH.

### **5. Run the Simulation**
To train the DQN model, use the `training.py` script:
```bash
python training.py
```

---

## **Usage**

### **Input Files**
The project requires:
- A SUMO-compatible road network file (e.g., `osm.net.xml`).
- A route file specifying traffic flows (e.g., `osm.rou.xml`).

### **Output Files**
- **Training Logs**: TensorBoard logs stored in the `Training/TensorBoard` directory.
- **Model Checkpoints**: Saved in `Training/Saved_Models`.
- **Simulation Data**: CSV outputs stored in `Training/CSV_Outputs`.

### **TensorBoard Visualization**
To monitor training progress:
```bash
tensorboard --logdir=Training/TensorBoard
```
Open the provided URL in your browser.

---

## **Project Structure**
```
TrafficSignalOptimization/
├── PALO ALTO/                 # Road network files (e.g., osm.net.xml, osm.rou.xml)
├── Training/
│   ├── CSV_Outputs/           # Simulation output files
│   ├── Saved_Models/          # Saved models and checkpoints
│   ├── TensorBoard/           # TensorBoard logs
├── training.py                # Training script for the DQN agent
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## **Key Results**
- The RL agent significantly outperformed the baseline system in high-traffic scenarios, reducing waiting times and congestion.
- Comparative performance graphs can be found in the repository.

---

## **Future Scope**
- Testing the model on multiple road networks to improve generalization.
- Exploring advanced RL algorithms like PPO or A3C.
- Incorporating real-world traffic data for enhanced realism.

---

## **References**

1. SUMO Documentation: [SUMO](https://sumo.dlr.de/docs/)
2. Alegre, L., & Bazzan, A. (2020). SUMO-RL: A reinforcement learning library for traffic signal control in SUMO. [GitHub repository](https://github.com/LucasAlegre/sumo-rl)
3. Nicholas Renotte. "Reinforcement Learning in 3 Hours | Full Course Using Python." [YouTube](https://www.youtube.com/watch?v=Mut_u40Sqz4)
4. TensorBoard Documentation: [TensorBoard](https://www.tensorflow.org/tensorboard)
5. Gymnasium Documentation: [Gymnasium](https://gymnasium.farama.org/)
6. NetEdit: [NetEdit](https://sumo.dlr.de/docs/Netedit/index.html)
7. OSM Web Wizard: [OSM Web Wizard](https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html)
