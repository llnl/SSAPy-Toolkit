import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;

public final class OrekitTwoBody {
    private OrekitTwoBody() {}

    public static void main(String[] args) {
        final double mu = Double.parseDouble(args[0]);
        final double radius = Double.parseDouble(args[1]);
        final double duration = Double.parseDouble(args[2]);
        final double step = Double.parseDouble(args[3]);
        final double speed = Math.sqrt(mu / radius);
        final Frame frame = FramesFactory.getGCRF();
        final AbsoluteDate epoch = AbsoluteDate.J2000_EPOCH;
        final CartesianOrbit initial = new CartesianOrbit(
            new PVCoordinates(
                new Vector3D(radius, 0.0, 0.0),
                new Vector3D(0.0, speed, 0.0)),
            frame, epoch, mu);
        final KeplerianPropagator propagator = new KeplerianPropagator(initial);

        System.out.println("t_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s");
        for (double elapsed = 0.0; elapsed <= duration + 0.5 * step; elapsed += step) {
            final SpacecraftState state = propagator.propagate(epoch.shiftedBy(elapsed));
            final PVCoordinates pv = state.getPVCoordinates(frame);
            final Vector3D r = pv.getPosition();
            final Vector3D v = pv.getVelocity();
            System.out.printf(
                java.util.Locale.ROOT,
                "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g%n",
                elapsed, r.getX(), r.getY(), r.getZ(),
                v.getX(), v.getY(), v.getZ());
        }
    }
}
