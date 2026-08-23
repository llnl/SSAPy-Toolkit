6-DoF Architecture Study
========================

Purpose
-------

This document records the design rationale for six-degree-of-freedom (6-DoF)
spacecraft dynamics in SSAPy Toolkit (SSATK). It is a trade study, not a
benchmark, endorsement, or complete review of every flight-dynamics package.
The external tools listed here were reviewed to identify common modeling
patterns, API boundaries, and non-goals for SSATK.

Survey date: 2026-08-20.

Problem Statement
-----------------

SSATK already depends on SSAPy for high-fidelity orbit modeling. The 6-DoF
layer should extend that orbit workflow with spacecraft attitude and body
physics without replacing SSAPy, Basilisk, Tudat, GMAT, Orekit, STK, FreeFlyer,
or launch-vehicle flight-dynamics tools.

The target use case is an analyst who already works with SSAPy ``Orbit`` objects
or inertial position/velocity arrays and needs to add:

* quaternion attitude state,
* body-frame angular rate,
* spacecraft mass and inertia,
* attitude-dependent drag and solar-radiation-pressure forces,
* body-frame torques,
* finite thrust and mass depletion,
* optional hardware components such as facets, tanks, thrusters, and wheels.

The 6-DoF layer should keep the same practical style as the rest of SSATK:
small Python objects, direct NumPy arrays, explicit units, and examples that can
be read as documentation.

External Tools Reviewed
-----------------------

NASA 42
^^^^^^^

NASA 42 is a direct reference for general-purpose spacecraft attitude and orbit
dynamics. Its public README describes multi-body spacecraft attitude dynamics,
rigid and flexible bodies, concurrent spacecraft, contact forces, two-body and
three-body orbital regimes, and visualization.

Design lesson for SSATK:

* Support a rigid-body hub first.
* Keep the API open to multiple bodies and flexible appendages later.
* Do not require a full simulation framework for simple concept studies.

Reference: https://github.com/ericstoneking/42

Basilisk
^^^^^^^^

Basilisk is the most relevant open-source architecture reference. Its
``spacecraft`` module is built around a hub plus attached ``stateEffectors`` and
``dynamicEffectors``. Public Basilisk docs describe reaction wheels, fuel tanks,
thrusters, hinged rigid bodies, radiation pressure, facet drag, and faceted
spacecraft models.

Design lesson for SSATK:

* Separate the spacecraft state from the force/torque providers.
* Treat hardware as components attached to a central spacecraft body.
* Distinguish ``state effectors`` that add integrated states from ``dynamic
  effectors`` that only contribute forces and torques.
* Start with rigid, fixed components; add articulated/flexible states only when
  a real SSATK workflow needs them.

Reference: https://github.com/AVSLab/basilisk

Tudat and TudatPy
^^^^^^^^^^^^^^^^^

Tudat/TudatPy provides the strongest astrodynamics modeling pattern for
combining translational, rotational, and mass propagation. Its documentation
exposes translational, rotational, mass, multi-type, multi-arc, torque,
aerodynamic, and radiation-pressure setup objects. Its force and torque
interfaces distinguish environmental models from propagated state settings.

Design lesson for SSATK:

* Keep translational, rotational, and mass states conceptually separate even
  when one integrator propagates them together.
* Make the force and torque model interface composable.
* Allow later multi-arc or estimation use without forcing that complexity into
  the first public API.

Reference: https://github.com/tudat-team/tudatpy

NASA JEOD and Trick
^^^^^^^^^^^^^^^^^^^

NASA JSC Engineering Orbital Dynamics (JEOD) is designed for use with the NASA
Trick Simulation Environment. Its public README describes environment,
dynamics, interaction, and utility models; standalone spacecraft trajectory and
attitude state; and coupling to effectors and guidance, navigation, and control
systems.

Design lesson for SSATK:

* Keep environment models, dynamics models, interaction models, and utilities
  separable.
* Do not make SSATK a general simulation executive. Python function calls and
  explicit object composition are enough for this package.

References:

* https://github.com/nasa/jeod
* https://github.com/nasa/trick

GMAT
^^^^

NASA GMAT is a mission-design and orbit-analysis reference, especially for
finite burns, tanks, thrusters, SPAD files, and operational maneuver workflows.
GMAT's torque-modeling documentation states that current torque outputs are
available for reporting and are not included in attitude propagation.

Design lesson for SSATK:

* Reuse the useful operational concepts: spacecraft hardware, tanks, thrusters,
  finite burns, and force-model reports.
* Do not treat GMAT as the architecture reference for coupled 6-DoF attitude
  propagation.

Reference: https://github.com/nasa/GMAT

Orekit
^^^^^^

Orekit is a mature astrodynamics library with strong force-model, maneuver,
attitude-provider, frame, and event-handling architecture. Its public docs
describe force models that access spacecraft state, attitude, mass, date, and
frame, plus impulse and continuous-thrust maneuvers. Orekit is not primarily a
spacecraft rigid-body dynamics simulator.

Design lesson for SSATK:

* Use explicit force-model interfaces.
* Preserve event and maneuver concepts for future finite-burn and control
  development.
* Keep attitude providers distinct from attitude dynamics.

Reference: https://www.orekit.org/

FreeFlyer and STK/Astrogator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FreeFlyer and STK/Astrogator are operational mission-design references for
finite burns, attitude reference frames, maneuver targeting, and analyst-facing
workflows. Public FreeFlyer documentation describes finite burns with active
thrusters, tank mass depletion, burn duration, burn reference frames, and thrust
steering. These tools are not the best model for a lightweight open Python
6-DoF implementation.

Design lesson for SSATK:

* Use user-facing terms familiar to mission analysts: finite burn, burn frame,
  tank, thruster, thrust direction, and mass depletion.
* Keep the first SSATK implementation scriptable and inspectable instead of
  reproducing a full mission-control-sequence system.

References:

* https://ai-solutions.com/freeflyer/
* https://www.ansys.com/products/missions/ansys-stk

JSBSim, RocketPy, and OpenRocket
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JSBSim, RocketPy, and OpenRocket are useful references for atmospheric vehicle
and launch-vehicle 6-DoF modeling. They emphasize configurable aerodynamics,
propulsion, atmosphere, and vehicle geometry. Their core problem is different
from SSATK's orbital and cislunar spacecraft focus.

Design lesson for SSATK:

* Borrow clean body/component/aerodynamic patterns where useful.
* Do not tune SSATK's 6-DoF API around launch-vehicle-specific concepts unless
  a rocket workflow explicitly needs them.

References:

* https://github.com/JSBSim-Team/jsbsim
* https://github.com/RocketPy-Team/RocketPy
* https://github.com/openrocket/openrocket

Capability Matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 18 24

   * - Tool
     - Coupled 6-DoF
     - Spacecraft body model
     - Maneuver hardware
     - Best lesson for SSATK
   * - NASA 42
     - Yes
     - Rigid/flexible multi-body
     - Thrusters and FSW concepts
     - Keep the path open to multi-body and flexible dynamics.
   * - Basilisk
     - Yes
     - Hub plus effectors
     - Thrusters, tanks, wheels
     - Use componentized state and dynamic effectors.
   * - Tudat/TudatPy
     - Yes
     - Environment/body settings
     - Thrust and mass propagation
     - Separate translational, rotational, mass, force, and torque settings.
   * - JEOD/Trick
     - Yes
     - Simulation-framework components
     - Effectors via simulation coupling
     - Keep environment, dynamics, and interactions modular.
   * - GMAT
     - Partial
     - Spacecraft hardware and SPAD
     - Strong finite-burn workflow
     - Reuse operational concepts, not attitude-dynamics architecture.
   * - Orekit
     - Partial
     - State, mass, attitude providers
     - Strong maneuver/event model
     - Keep force models explicit and event-ready.
   * - FreeFlyer/STK
     - Partial
     - Analyst-facing spacecraft objects
     - Strong targeting workflows
     - Use familiar user terms and burn-frame concepts.
   * - JSBSim/RocketPy/OpenRocket
     - Yes, atmospheric focus
     - Configurable vehicle geometry
     - Propulsion and aero
     - Borrow geometry patterns, avoid launch-specific API bias.

Chosen SSATK Architecture
-------------------------

SSATK should implement a small, composable 6-DoF layer with these boundaries:

``Spacecraft``
    A user-facing state object with inertial position, inertial velocity,
    epoch, body-to-inertial quaternion, body-frame angular rate, mass, inertia,
    and optional attached components.

``SpacecraftBody`` or ``RigidBody``
    A physical body definition containing mass properties, center of mass,
    reference geometry, and body-frame component locations. This should become
    the source of truth for mass, inertia, reference area, center of pressure,
    and visual geometry.

``Facet``
    A fixed body-frame surface element with area, normal, center of pressure,
    drag coefficient, reflectivity coefficient, and optional optical properties.
    The first implementation should support fixed facets only.

``Thruster``
    A body-frame actuator with location, thrust direction, thrust magnitude or
    thrust schedule, specific impulse, and optional tank connection.

``Tank``
    A mass component with propellant mass, dry mass or hardware mass, body-frame
    location, and optional inertia contribution. The first implementation can
    use constant inertia or simple point-mass parallel-axis updates.

``ReactionWheel``
    A body-frame actuator with a torque axis, torque limit, and optional
    momentum-capacity metadata. The current SSATK implementation contributes
    saturated body-frame torque; separate wheel-speed/momentum-state
    propagation remains a later extension.

``Environment``
    A provider for atmosphere density, wind, Sun position, third-body positions,
    eclipse fraction, and gravity constants. This keeps physical environment
    data out of individual force models.

``ForceTorqueModel``
    A callable object that receives the spacecraft state, body model, and
    environment, then returns inertial force, body-frame torque, and optional
    mass-flow contributions.

Near-Term Implementation Plan
-----------------------------

The practical first release should be deliberately limited:

1. Keep the existing ``Spacecraft`` and ``propagate_6dof`` user workflow.
2. Add a body-model object that can be attached to ``Spacecraft``.
3. Add fixed-facet geometry and one helper constructor for a box-wing
   spacecraft.
4. Convert flat-plate drag and solar-radiation-pressure models to operate over
   one or more facets.
5. Add thruster components with body-frame force, body-frame torque, and
   optional mass-flow output.
6. Add reaction wheels and a basic quaternion PD controller for simple attitude
   studies.
7. Add tests that check force direction, torque sign, quaternion normalization,
   mass-flow sign, and agreement with the existing single-flat-plate
   functions for the one-facet case.

Non-Goals for the First SSATK 6-DoF Release
-------------------------------------------

These features are useful, but should not be required for the initial
architecture:

* flexible bodies,
* fuel slosh,
* articulated solar arrays,
* reaction-wheel momentum-state propagation,
* contact dynamics,
* high-order geopotential torque,
* full finite-element or CAD-derived spacecraft geometry,
* a simulation executive,
* a GUI or mission-control-sequence language.

These can be added later without breaking the first API if the body, component,
environment, and force/torque boundaries remain clean.

Current SSATK State
-------------------

The current SSATK 6-DoF implementation already has the correct minimal
numerical backbone:

* ``ssapy_toolkit.dynamics.Spacecraft`` stores inertial position, inertial
  velocity, quaternion attitude, angular rate, inertia, mass, area, drag
  coefficient, reflectivity coefficient, and center of pressure.
* ``ssapy_toolkit.dynamics.propagate_6dof`` propagates position, velocity,
  quaternion, and angular rate with user-provided acceleration and torque
  models.
* ``ssapy_toolkit.accelerations_6dof`` contains point-mass gravity, J2,
  third-body gravity, cannonball drag, cannonball solar-radiation pressure,
  constant inertial/NTW/body acceleration, summed force/torque helpers, and
  flat-plate/facet drag/SRP, thruster, magnetic-dipole, reaction-wheel, and
  quaternion PD attitude-control torque models.
* ``ssapy_toolkit.accelerations_6dof.SpacecraftManeuverAccel`` adds finite
  maneuver acceleration with explicit ``frame`` selection and scalar, callable,
  analytical, or CSV-loaded thrust curves.
* ``ssapy_toolkit.dynamics.attitude_quaternion_from_frame`` converts existing
  SSATK satellite-operation frame definitions, such as ``ntw``, ``vnb``, and
  ``nadir_velocity``, into body-to-GCRF target quaternions for attitude-control
  studies.

The missing layer is not the integrator. The remaining major gaps are higher
fidelity body/component state propagation: wheel momentum states, articulated
appendages, flexible bodies, propellant slosh, and event-driven finite-burn
segments.

Recommended Direction
---------------------

SSATK should follow Basilisk's componentized spacecraft pattern and Tudat's
separation of propagated states, environment, forces, and torques, while
remaining smaller and more direct than either package.

SSAPy's ``NTW`` convention remains useful because it is already part of the
SSAPy API and gives a compact normal/tangential/cross-track burn input. It is
not the only standard used in satellite operations. Mission-analysis tools also
use ``RTN``/``RSW``/``RIC``/``LVLH`` for radial-transverse-normal commands and
``VNB``/``VNC`` for velocity-normal-binormal commands. SSATK should therefore
avoid a new NTW-only interface and expose a single ``frame=...`` argument. This
keeps SSAPy compatibility through ``frame="ntw"`` while making operational
frame choices explicit.

The public workflow should remain:

.. code-block:: python

   import numpy as np
   import ssapy_toolkit as ssatk

   spacecraft = ssatk.Spacecraft(
       r=[7_000_000.0, 0.0, 0.0],
       v=[0.0, 7_500.0, 0.0],
       q=[1.0, 0.0, 0.0, 0.0],
       omega=[0.0, 0.0, 0.001],
       mass=100.0,
       inertia=np.diag([10.0, 12.0, 8.0]),
   )

   body = ssatk.satellite_design(
       "earth_observation",
       mass=500.0,
       solar_array_area=10.0,
   ).with_components(
       ssatk.Component(mass=25.0, position_body=[0.0, 0.0, 0.7], name="payload"),
   )

   trajectory = ssatk.Spacecraft(r=spacecraft.r, v=spacecraft.v, body=body).propagate(
       times=np.linspace(0.0, 600.0, 61),
       models=[
           ssatk.SpacecraftAccelJ2(),
           ssatk.SpacecraftFacetDrag(density=1e-12),
           ssatk.SpacecraftFacetSolRad([ssatk.AU, 0.0, 0.0]),
       ],
   )

This keeps SSATK aligned with SSAPy users while adding real body physics in a
way that can grow toward higher-fidelity spacecraft modeling.
