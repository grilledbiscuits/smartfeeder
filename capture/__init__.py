"""Motion-triggered capture pipeline for the feeder camera.

Third sibling to `ml/` (the classifier) and `web/` (the dashboard), and the
only component that touches hardware. It imports from both:

* `birdcam.inference.Classifier` turns model outputs into one honest label.
  This package never re-implements that decision -- it feeds it frames and
  reads `Decision.should_record`.
* `web.db.add_visit` plus `web.paths` is the dashboard's stated interface
  contract (see web/db.py). Publishing a visit means writing the media into
  var/media/ and inserting one row. There is no upload endpoint and this
  package does not add one.

Nothing here is importable from `birdcam` or `web`. The dependency runs one
way, so the dashboard can be stopped or the ML package re-exported without
this service noticing.
"""

__version__ = "0.1.0"
