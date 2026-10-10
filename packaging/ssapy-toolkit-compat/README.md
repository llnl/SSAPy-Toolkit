# ssapy-toolkit → ssatk

The Space Situational Awareness Toolkit was published as `ssapy-toolkit` and
imported as `ssapy_toolkit` through 1.0.7. From 1.1.0 it is published as
[`ssatk`](https://pypi.org/project/ssatk/) and imported as `ssatk`.

This package installs `ssatk` and a small `ssapy_toolkit` alias so existing
code keeps running; importing `ssapy_toolkit` emits a `FutureWarning`. New
code should use:

```bash
python -m pip install ssatk
```

```python
import ssatk
```
