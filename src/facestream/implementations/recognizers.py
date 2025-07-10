import numpy as np
import tensorflow as tf
from ..components import FaceRecognizer
import cv2
import insightface
from insightface.app import FaceAnalysis

class FaceNetRecognizer(FaceRecognizer):
    """
    A face recognizer that uses the FaceNet model to compute embeddings.
    """
    def __init__(self, model_path: str):
        """
        Initializes the FaceNet recognizer.

        Args:
            model_path: The path to the FaceNet model file.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        
        self.input_tensor = self.graph.get_tensor_by_name('input:0')
        self.output_tensor = self.graph.get_tensor_by_name('embeddings:0')
        self.phase_train_tensor = self.graph.get_tensor_by_name('phase_train:0')

    def compute_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Computes a feature vector (embedding) for a face image.

        Args:
            face_crop: The face image as a NumPy array.

        Returns:
            The computed embedding as a NumPy array.
        """
        # Preprocess the face crop
        face_crop = cv2.resize(face_crop, (160, 160))
        face_crop = (face_crop - 127.5) / 128.0
        face_crop = np.expand_dims(face_crop, axis=0)

        with tf.compat.v1.Session(graph=self.graph) as sess:
            feed_dict = {
                self.input_tensor: face_crop,
                self.phase_train_tensor: False
            }
            embedding = sess.run(self.output_tensor, feed_dict=feed_dict)[0]
        return embedding

class InsightFaceRecognizer(FaceRecognizer):
    """
    A face recognizer that uses the InsightFace model to compute embeddings.
    """
    def __init__(self, model_path: str = 'buffalo_l'):
        """
        Initializes the InsightFace recognizer.

        Args:
            model_path: The name of the InsightFace model to use (e.g., 'buffalo_l').
        """
        self.app = FaceAnalysis(name=model_path, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def compute_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Computes a feature vector (embedding) for a face image using InsightFace.

        Args:
            face_crop: The face image as a NumPy array.

        Returns:
            The computed embedding as a NumPy array.
        """
        # InsightFace's app.get() expects a full image, not just a crop.
        # However, since we are passing a face_crop, we assume it's already
        # a cropped image containing a single face.
        # We need to ensure the face_crop is in the correct format (BGR).
        
        # The app.get() method returns a list of 'Face' objects.
        faces = self.app.get(face_crop)

        if len(faces) > 0:
            # We'll work with the first detected face in the crop.
            # In a real scenario, if multiple faces are detected in a crop,
            # you might need a strategy to pick the correct one.
            first_face = faces[0]
            embedding = first_face.normed_embedding
            return embedding
        else:
            # If no face is detected in the crop, return an empty array or raise an error
            # For now, returning an array of zeros as a placeholder
            return np.zeros(512).astype(np.float32)