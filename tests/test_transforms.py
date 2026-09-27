import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from dataset_utils.transforms import Augmentation


class SynchronizedZoomOutTest(unittest.TestCase):
    def test_same_geometry_is_used_for_every_frame_and_target(self):
        transform = Augmentation(
            img_size=224,
            zoom_out_prob=1.0,
            zoom_out_min_scale=0.5,
        )
        frame = Image.fromarray(np.full((224, 224, 3), 200, dtype=np.uint8))

        with patch("dataset_utils.transforms.random.random", return_value=0.0), \
                patch("dataset_utils.transforms.random.uniform", return_value=0.5), \
                patch(
                    "dataset_utils.transforms.random.randint",
                    side_effect=[10, 20],
                ):
            frames, params = transform.random_zoom_out([frame, frame.copy()])

        self.assertTrue(np.array_equal(np.asarray(frames[0]), np.asarray(frames[1])))
        self.assertEqual(params, (0.5, 10 / 224, 20 / 224))

        target_a = np.array([[0.25, 0.25, 0.75, 0.75, 1.0]])
        target_b = target_a.copy()
        transformed_a = transform.apply_zoom_out_bbox(target_a, params)
        transformed_b = transform.apply_zoom_out_bbox(target_b, params)

        np.testing.assert_allclose(transformed_a, transformed_b)
        self.assertAlmostEqual(transformed_a[0, 2] - transformed_a[0, 0], 0.25)
        self.assertAlmostEqual(transformed_a[0, 3] - transformed_a[0, 1], 0.25)
        self.assertTrue(np.all((transformed_a[:, :4] >= 0.0)))
        self.assertTrue(np.all((transformed_a[:, :4] <= 1.0)))


if __name__ == "__main__":
    unittest.main()
