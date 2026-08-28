import java.io.File;
import java.util.Locale;
import java.util.regex.Pattern;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.orekit.data.DataProvidersManager;
import org.orekit.data.DirectoryCrawler;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.ICGEMFormatReader;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.OrbitType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;

public final class OrekitJ2 {
    private OrekitJ2() {}

    public static void main(String[] args) {
        final File coefficientFile = new File(args[0]);
        final double duration = Double.parseDouble(args[1]);
        final double step = Double.parseDouble(args[2]);
        final DataProvidersManager data = DataProvidersManager.getInstance();
        data.clearProviders();
        data.addProvider(new DirectoryCrawler(coefficientFile.getParentFile()));
        GravityFieldFactory.clearPotentialCoefficientsReaders();
        GravityFieldFactory.addPotentialCoefficientsReader(
            new ICGEMFormatReader(Pattern.quote(coefficientFile.getName()), false));

        final Frame frame = FramesFactory.getGCRF();
        final AbsoluteDate epoch = AbsoluteDate.J2000_EPOCH;
        final NormalizedSphericalHarmonicsProvider provider =
            GravityFieldFactory.getConstantNormalizedProvider(2, 0);
        final double mu = provider.getMu();
        final double ae = provider.getAe();
        final double c20 = provider.onDate(epoch).getNormalizedCnm(2, 0);
        final double radius = 7_000_000.0;
        final double speed = Math.sqrt(mu / radius);
        final double inclination = Math.toRadians(63.0);
        final CartesianOrbit initial = new CartesianOrbit(
            new PVCoordinates(
                new Vector3D(radius, 0.0, 0.0),
                new Vector3D(0.0, speed * Math.cos(inclination), speed * Math.sin(inclination))),
            frame, epoch, mu);
        final double[][] tolerances = NumericalPropagator.tolerances(
            1.0e-3, initial, OrbitType.CARTESIAN);
        final DormandPrince853Integrator integrator = new DormandPrince853Integrator(
            1.0e-3, step, tolerances[0], tolerances[1]);
        final NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setOrbitType(OrbitType.CARTESIAN);
        propagator.setInitialState(new SpacecraftState(initial));
        propagator.addForceModel(new HolmesFeatherstoneAttractionModel(frame, provider));

        System.out.printf(Locale.ROOT, "# mu=%.17g,ae=%.17g,c20=%.17g%n", mu, ae, c20);
        System.out.println("t_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s");
        for (double elapsed = 0.0; elapsed <= duration + 0.5 * step; elapsed += step) {
            final PVCoordinates pv = propagator.propagate(epoch.shiftedBy(elapsed)).getPVCoordinates(frame);
            final Vector3D r = pv.getPosition();
            final Vector3D v = pv.getVelocity();
            System.out.printf(
                Locale.ROOT,
                "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g%n",
                elapsed, r.getX(), r.getY(), r.getZ(), v.getX(), v.getY(), v.getZ());
        }
    }
}
