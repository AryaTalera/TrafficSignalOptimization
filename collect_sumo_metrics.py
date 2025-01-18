import os
import sys
import traci
import matplotlib.pyplot as plt
import csv
from pathlib import Path

class SUMOMetricsCollector:
    def __init__(self, net_file, route_file, traffic_light_id, simulation_steps=2500, gui=False):
        self.net_file = net_file
        self.route_file = route_file
        self.traffic_light_id = traffic_light_id
        self.simulation_steps = simulation_steps
        self.gui = gui
        self.total_stopped_vehicles = []
        self.avg_waiting_times = []

    def validate_files(self):

        if not all(Path(f).exists() for f in [self.net_file, self.route_file]):
            raise FileNotFoundError("Network or route file not found")
        
        if 'SUMO_HOME' not in os.environ:
            raise EnvironmentError("Please set SUMO_HOME environment variable")
        
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        if tools not in sys.path:
            sys.path.append(tools)

    def initialize_simulation(self):

        self.validate_files()
        
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary),
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log",
            "--no-warnings",
        ]
        
        traci.start(sumo_cmd)
        traci.trafficlight.setPhase(self.traffic_light_id, 0)

    def get_system_info(self):

        vehicles = traci.vehicle.getIDList()
        
        # Calculate total stopped vehicles (speed < 0.1 m/s)
        stopped = len([v for v in vehicles if traci.vehicle.getSpeed(v) < 0.1])
        
        # Calculate mean waiting time across all vehicles
        waiting_time = 0.0
        for v in vehicles:
            waiting_time += traci.vehicle.getAccumulatedWaitingTime(v)
        
        mean_waiting_time = waiting_time / len(vehicles) if vehicles else 0.0
        
        return stopped, mean_waiting_time

    def collect_metrics(self):

        for step in range(self.simulation_steps):
            traci.simulationStep()
            
            stopped_vehicles, mean_waiting_time = self.get_system_info()
            
            self.total_stopped_vehicles.append(stopped_vehicles)
            self.avg_waiting_times.append(mean_waiting_time)
            
            print(f"Step: {step}, Stopped Vehicles: {stopped_vehicles}, "
                  f"Avg Waiting Time: {mean_waiting_time:.2f}")

    def save_metrics(self, filename='sumo_metrics.csv'):

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Stopped Vehicles", "Avg Waiting Time"])
            for step, (vehicles, waiting_time) in enumerate(
                zip(self.total_stopped_vehicles, self.avg_waiting_times)):
                writer.writerow([step, vehicles, waiting_time])

    def plot_results(self):

        plt.figure(figsize=(12, 6))

        # Total stopped vehicles plot
        plt.subplot(1, 2, 1)
        plt.plot(self.total_stopped_vehicles, 
                 label='Total Stopped Vehicles')
        plt.xlabel('Step')
        plt.ylabel('Total Stopped Vehicles')
        plt.title('Total Stopped Vehicles Over Time')
        plt.legend()

        # Average waiting time plot
        plt.subplot(1, 2, 2)
        plt.plot(self.avg_waiting_times, 
                 label='Average Waiting Time', 
                 color='orange')
        plt.xlabel('Step')
        plt.ylabel('Average Waiting Time (s)')
        plt.title('Average Waiting Time Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def run(self):

        try:
            self.initialize_simulation()
            self.collect_metrics()
            self.save_metrics()
            self.plot_results()
        finally:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass

if __name__ == "__main__":
    NET_FILE = 'C:/Programming/BYOP/PALO ALTO/osm.net.xml'
    ROUTE_FILE = 'C:/Programming/BYOP/PALO ALTO/osm.rou.xml'
    TRAFFIC_LIGHT_ID = "cluster_2508688051_5148642229_5148642231"
    
    collector = SUMOMetricsCollector(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        traffic_light_id=TRAFFIC_LIGHT_ID,
        simulation_steps=2500,
        gui=True
    )
    collector.run()