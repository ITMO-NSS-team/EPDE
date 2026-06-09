import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import MinMaxScaler

# ==========================
# === DATA PROCESSOR CLASS ===
# ==========================
class DataProcessor:
    """
    Class for preprocessing raw trajectory data.

    Responsibilities:
    ----------------
    - Extract structured coordinate and angle data from raw input
    - Normalize coordinates into a common scale
    - Store and manage processed trajectory data

    Attributes:
    ----------
    data : List
        Raw input data. Expected format:
        [
            [
                (id, angle_in_degrees, [x, y]),
                ...
            ],
            ...
        ]

    coord_data : Dict[int, List[List[float]]]
        Dictionary mapping object ID → list of (x, y) coordinates

    angle_data : Dict[int, List[float]]
        Dictionary mapping object ID → list of angles (in radians)

    normalized_coord_data : Dict[int, np.ndarray]
        Dictionary mapping object ID → normalized trajectory (Nx2 array)

    scaler_x, scaler_y : MinMaxScaler
        Scalers used for normalization of x and y coordinates
    """

    def __init__(self, data: List):
        self.data = data
        self.coord_data = {}
        self.angle_data = {}
        self.normalized_coord_data = {}

    # === Data extraction ===
    def extract_data(self) -> Tuple[Dict, Dict]:
        """
        Extract coordinate and angle data from raw input.

        Converts:
        - angles from degrees to radians
        - groups data by object ID
        """
        coord_data = {}
        angle_data = {}

        for inner_list in self.data:
            for item in inner_list:
                key = item[0]
                angle = np.deg2rad(item[1]) 
                coordinates = item[2]

                if key not in coord_data:
                    coord_data[key] = []
                    angle_data[key] = []

                coord_data[key].append(coordinates)
                angle_data[key].append(angle)

        self.coord_data = coord_data
        self.angle_data = angle_data
        return coord_data, angle_data

    # === Normalization ===
    @staticmethod
    def normalize_to_normal_distribution(data: np.ndarray) -> np.ndarray:
        """
        Normalize 1D data to [0, 1] range using MinMax scaling.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        return normalized_data, scaler

    def normalize_all_coordinates(self) -> Dict:
        """
        Normalize all trajectory coordinates across all objects.

        Workflow:
        --------
        1. Collect all x and y values globally
        2. Fit MinMaxScaler separately for x and y
        3. Apply normalization
        4. Reconstruct per-object trajectories
        """
        all_x, all_y = [], []
        
        for points in self.coord_data.values():
            points_array = np.array(points)
            all_x.extend(points_array[:, 0])
            all_y.extend(points_array[:, 1])

        x_normalized, scaler_x = self.normalize_to_normal_distribution(np.array(all_x))
        y_normalized, scaler_y = self.normalize_to_normal_distribution(np.array(all_y))

        self.scaler_x,  self.scaler_y = scaler_x, scaler_y

        idx = 0
        for obj_id, points in self.coord_data.items():
            points_array = np.array(points)
            num_points = len(points_array)
            obj_x_norm = x_normalized[idx:idx + num_points]
            obj_y_norm = y_normalized[idx:idx + num_points]
            self.normalized_coord_data[obj_id] = np.column_stack((obj_x_norm, obj_y_norm))
            idx += num_points

        return self.normalized_coord_data
    
    def transform_new_robot(self, x_new, y_new, id_new):
        """
        Apply existing normalization to a new robot (particle) trajectory.

        Important:
        ----------
        Uses already fitted scalers (scaler_x, scaler_y),
        so must be called AFTER normalize_all_coordinates().

        Parameters:
        ----------
        x_new : np.ndarray
            New robot x-coordinates

        y_new : np.ndarray
            New robot y-coordinates

        id_new : int
            Identifier for the new robot

        Returns:
        -------
        normalized_coord_data : Dict[int, np.ndarray]
            Updated dictionary including new robot

        key : int
            ID of the added robot
        """
        x_scaled = self.scaler_x.transform(x_new.reshape(-1,1))
        y_scaled = self.scaler_y.transform(y_new.reshape(-1,1))

        key = id_new

        self.normalized_coord_data[key] = np.column_stack((x_scaled, y_scaled))

        return self.normalized_coord_data, key
