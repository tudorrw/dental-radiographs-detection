import numpy as np

class ToothLabelMapper:
    def __init__(self):
        """
        Initializes the label mapping by defining the valid tooth labels.
        The labels are structured according to the FDI numbering system.
        """
        self.tooth_labels = self._generate_labels()
        self.label_to_index = {label: idx for idx, label in enumerate(self.tooth_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def _generate_labels(self):
        """
        Generates the structured list of tooth labels while ensuring proper order.
        
        Returns:
            list: A list of valid tooth labels.
        """
        background = [0]  # Background class
        quadrant1 = (np.arange(1, 9) + 10)[::-1]  # Quadrant 1 (upper left) (Teeth 18 → 11) teeth from 9 to 16
        quadrant2 = np.arange(1, 9)               # Quadrant 2 (upper right) (Teeth 21 → 28) teeth from 8 to 1
        quadrant3 = (np.arange(1, 9) + 30)[::-1]  # Quadrant 3 (lower right) (Teeth 48 → 41) teeth from 25 to 32
        quadrant4 = np.arange(1, 9) + 20          # Quadrant 4 (lower left) (Teeth 31 → 38) teeth from 24 to 17

        return background + quadrant1.tolist() + quadrant2.tolist() + quadrant3.tolist() + quadrant4.tolist()

    def encode(self, labels):
        """
        Converts a list of FDI-based tooth labels into indexed class labels.

        Args:
            labels (list): List of original tooth labels.

        Returns:
            np.ndarray: Transformed labels mapped to their corresponding class indices.
        """
        return np.array([self.label_to_index[label] for label in labels])

    def decode(self, encoded_labels):
        """
        Converts indexed class labels back to their original FDI tooth labels.

        Args:
            encoded_labels (list or np.ndarray): List of encoded class indices.

        Returns:
            np.ndarray: Transformed labels mapped back to their original FDI numbers.
        """
        return np.array([self.index_to_label[label] for label in encoded_labels])
