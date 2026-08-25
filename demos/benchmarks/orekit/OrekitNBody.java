import java.io.File;
import java.util.Locale;

import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.data.DataProvidersManager;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.OrbitType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.PVCoordinates;
import org.orekit.forces.gravity.ThirdBodyAttraction;
import org.hipparchus.geometry.euclidean.threed.Vector3D;

public final class OrekitNBody {
    private OrekitNBody() {}

    public static void main(String[] args) {
        final String dataDirectory = args[0];
        final String mode = args[1];
        final double mu = Double.parseDouble(args[2]);
        final double radius = Double.parseDouble(args[3]);
        final double duration = Double.parseDouble(args[4]);
        final double step = Double.parseDouble(args[5]);
        final String epochText = args[6];

        final DataProvidersManager data = DataProvidersManager.getInstance();
        data.clearProviders();
        data.addProvider(new DirectoryCrawler(new File(dataDirectory)));
        CelestialBodyFactory.clearCelestialBodyLoaders();
        CelestialBodyFactory.clearCelestialBodyCache();

        final TimeScale utc = TimeScalesFactory.getUTC();
        final AbsoluteDate epoch = new AbsoluteDate(epochText, utc);
        final Frame frame = FramesFactory.getGCRF();
        final double speed = Math.sqrt(mu / radius);
        final CartesianOrbit initial = new CartesianOrbit(
            new PVCoordinates(
                new Vector3D(radius, 0.0, 0.0),
                new Vector3D(0.0, speed, 0.0)),
            frame, epoch, mu);

        final double[][] tolerances = NumericalPropagator.tolerances(
            1.0e-3, initial, OrbitType.CARTESIAN);
        final DormandPrince853Integrator integrator = new DormandPrince853Integrator(
            1.0e-3, step, tolerances[0], tolerances[1]);
        final NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setOrbitType(OrbitType.CARTESIAN);
        propagator.setInitialState(new SpacecraftState(initial));

        final String[] bodies = mode.equals("earth_moon_sun")
            ? new String[] {"Moon", "Sun"}
            : new String[] {"Moon", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"};
        for (String name : bodies) {
            final CelestialBody body = CelestialBodyFactory.getBody(name);
            propagator.addForceModel(new ThirdBodyAttraction(body));
        }

        System.out.println("t_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s");
        for (double elapsed = 0.0; elapsed <= duration + 0.5 * step; elapsed += step) {
            final SpacecraftState state = propagator.propagate(epoch.shiftedBy(elapsed));
            final PVCoordinates pv = state.getPVCoordinates(frame);
            final Vector3D r = pv.getPosition();
            final Vector3D v = pv.getVelocity();
            System.out.printf(
                Locale.ROOT,
                "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g%n",
                elapsed, r.getX(), r.getY(), r.getZ(),
                v.getX(), v.getY(), v.getZ());
        }
    }
}
