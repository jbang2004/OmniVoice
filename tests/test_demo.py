import unittest

from omnivoice.cli.demo import _build_generation_config


class DemoConfigTests(unittest.TestCase):
    def test_build_generation_config_accepts_gradio_slider_values(self):
        config = _build_generation_config(
            num_step=32.0,
            guidance_scale=2,
            denoise=False,
            preprocess_prompt=True,
            postprocess_output=False,
        )

        self.assertEqual(config.num_step, 32)
        self.assertEqual(config.guidance_scale, 2.0)
        self.assertFalse(config.denoise)
        self.assertTrue(config.preprocess_prompt)
        self.assertFalse(config.postprocess_output)

    def test_build_generation_config_rejects_string_bool_values(self):
        with self.assertRaisesRegex(ValueError, "denoise must be bool"):
            _build_generation_config(
                num_step=32,
                guidance_scale=2.0,
                denoise="false",
                preprocess_prompt=True,
                postprocess_output=True,
            )

        with self.assertRaisesRegex(ValueError, "preprocess_prompt must be bool"):
            _build_generation_config(
                num_step=32,
                guidance_scale=2.0,
                denoise=True,
                preprocess_prompt="false",
                postprocess_output=True,
            )


if __name__ == "__main__":
    unittest.main()
