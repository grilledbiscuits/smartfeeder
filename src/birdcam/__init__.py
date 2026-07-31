"""birdcam -- South African nectarivore feeder-cam classifier.

Two prediction heads over a shared backbone:
  Head 1 (taxon)       -- species, with honest genus/family/guild fallback
  Head 2 (sex/plumage)  -- male_breeding / male_eclipse / female / juvenile /
                           indeterminate / not_applicable

Deployment target is a Raspberry Pi 5 + Hailo-8L running fully on-device.
"""

__version__ = "0.1.0"

from birdcam.config import Config, ConfigError, load_config

__all__ = ["Config", "ConfigError", "load_config", "__version__"]
