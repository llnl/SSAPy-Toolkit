Satellite lists: Space-Track and DISCOS
========================================

SSAPy Toolkit does not ship a satellite TLE catalog.  This keeps user data,
credentials, and provider terms out of source distributions.  Obtain data with
your own account and keep the resulting files in a user-local directory.

Space-Track TLEs
----------------

Create an account at https://www.space-track.org/auth/createAccount and accept
the current Space-Track terms.  Do not put credentials in a committed file.
Use environment variables, or create the ignored ``tle_credentials_local.py``
at the Toolkit repository root:

.. code-block:: python

   ST_USER = "you@example.com"
   ST_PASSWORD = "your-password"

Alternatively, set them in PowerShell:

.. code-block:: powershell

   $env:SPACETRACK_USER = "you@example.com"
   $env:SPACETRACK_PASSWORD = "your-password"

The updater writes the local list to ``ssapy_satellites.json`` and fresh TLEs
to ``tle_cache.json``.  By default these are under ``~/ssatk_data``.  Set
``SSATK_OUTPUT_DIR`` to choose another private directory:

.. code-block:: powershell

   $env:SSATK_OUTPUT_DIR = "D:\ssapy-private\satellites"
   python -m ssapy_toolkit.io.tle_updater --search "ISS" --max 10
   python -m ssapy_toolkit.io.tle_updater --add-group stations
   python -m ssapy_toolkit.io.tle_updater --fetch-all-active --max-active 1000
   python -m ssapy_toolkit.io.tle_updater --verify

The equivalent POSIX setup is:

.. code-block:: bash

   export SSATK_OUTPUT_DIR="$HOME/.local/share/ssapy/satellites"
   python -m ssapy_toolkit.io.tle_updater --search ISS --max 10

To use a list obtained by another approved Space-Track workflow, convert it
to a JSON array of objects containing ``name``, ``line1``, and ``line2`` (and,
when available, ``norad_id``), then save it as
``$SSATK_OUTPUT_DIR/ssapy_satellites.json``.  Run ``--verify`` before using it.
The list and cache are local working data; do not commit or redistribute them
unless Space-Track has granted the required permission.

ESA DISCOS object metadata
--------------------------

DISCOSweb is an ESA object metadata service, not a TLE provider.  Use it to
obtain object names, identifiers, and physical metadata, then use the
corresponding NORAD IDs with Space-Track for TLEs.

1. Register/sign in at https://discosweb.esoc.esa.int and accept ESA's current
   terms.
2. Create a personal access token at https://discosweb.esoc.esa.int/tokens/new.
3. Save it privately, or set ``DISCOS_TOKEN``.  The Toolkit helper can save it
   to the ignored local credential file:

   .. code-block:: bash

      python -m ssapy_toolkit.io.discos_client setup --save
      python -m ssapy_toolkit.io.discos_client login --accept-terms

4. Download a bounded object collection to a private path:

   .. code-block:: bash

      python -m ssapy_toolkit.io.discos_client download \
        --accept-terms --max-objects 500 \
        --output "$HOME/.local/share/ssapy/discos_objects.json"

   PowerShell users can use one line or PowerShell's backtick for continuation.
   Add ``--filter`` and ``--fields`` to limit the request.  The output includes
   provenance and required ESA attribution in both the JSON document and its
   ``.provenance.json`` sidecar.

Inspect a saved file without network access:

.. code-block:: bash

   python -m ssapy_toolkit.io.discos_client show \
     "$HOME/.local/share/ssapy/discos_objects.json" --flatten

Keep DISCOS downloads private unless the current ESA terms permit your intended
use.  Preserve the generated attribution and provenance when making derived
files.  Never place tokens or downloaded provider data in the repository.
