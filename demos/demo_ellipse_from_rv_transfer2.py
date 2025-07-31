import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, root_scalar
from scipy.linalg import norm
import warnings
warnings.filterwarnings('ignore')

class OrbitalTransferSolver:
    def __init__(self, mu=1.0):
        """
        Initialize the orbital transfer solver.
        mu: gravitational parameter (default 1.0 for normalized units)
        """
        self.mu = mu
        
    def orbital_elements_to_rv(self, a, e, i, Omega, omega, nu):
        """
        Convert orbital elements to position and velocity vectors.
        """
        # Semi-latus rectum
        p = a * (1 - e**2)
        
        # Radius
        r = p / (1 + e * np.cos(nu))
        
        # Position in perifocal coordinates
        r_pf = r * np.array([np.cos(nu), np.sin(nu), 0])
        
        # Velocity in perifocal coordinates
        v_pf = np.sqrt(self.mu / p) * np.array([
            -np.sin(nu),
            e + np.cos(nu),
            0
        ])
        
        # Rotation matrices
        R3_Omega = self._rotation_matrix_z(Omega)
        R1_i = self._rotation_matrix_x(i)
        R3_omega = self._rotation_matrix_z(omega)
        
        # Combined rotation matrix from perifocal to inertial
        Q = R3_Omega @ R1_i @ R3_omega
        
        # Transform to inertial frame
        r_vec = Q @ r_pf
        v_vec = Q @ v_pf
        
        return r_vec, v_vec
    
    def _rotation_matrix_x(self, angle):
        """Rotation matrix about x-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    def _rotation_matrix_z(self, angle):
        """Rotation matrix about z-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def lambert_solver(self, r1, r2, dt, direction=1):
        """
        Solve Lambert's problem using universal variables.
        Returns velocity vectors v1 and v2.
        """
        r1_mag = norm(r1)
        r2_mag = norm(r2)
        
        # Angle between position vectors
        cos_nu = np.dot(r1, r2) / (r1_mag * r2_mag)
        sin_nu = direction * np.sqrt(1 - cos_nu**2)
        
        # Fundamental parameters
        A = direction * np.sqrt(r1_mag * r2_mag * (1 + cos_nu))
        
        if abs(A) < 1e-10:
            return None, None
        
        # Initial guess for z
        z = 0
        
        # Newton-Raphson iteration
        for _ in range(50):
            C = self._stumpff_C(z)
            S = self._stumpff_S(z)
            
            y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)
            
            if y < 0:
                z *= 0.8
                continue
                
            x = np.sqrt(y / C)
            
            # Time equation
            t = (x**3 * S + A * np.sqrt(y)) / np.sqrt(self.mu)
            
            # Check convergence
            if abs(t - dt) < 1e-6:
                break
                
            # Derivatives for Newton-Raphson
            if z > 1e-10:
                Cp = (1 - z * S - 2 * C) / (2 * z)
                Sp = (C - 3 * S) / (2 * z)
            else:
                Cp = -1/24
                Sp = -1/120
                
            dt_dz = (x**3 * (Sp - 3 * S * Cp / (2 * C)) + 
                     A/8 * (3 * S * np.sqrt(y) / C + A / x)) / np.sqrt(self.mu)
            
            # Update z
            z -= (t - dt) / dt_dz
        
        # Calculate velocities
        f = 1 - y / r1_mag
        g = A * np.sqrt(y / self.mu)
        g_dot = 1 - y / r2_mag
        
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g
        
        return v1, v2
    
    def _stumpff_C(self, z):
        """Stumpff function C(z)"""
        if z > 0:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < 0:
            return (np.cosh(np.sqrt(-z)) - 1) / (-z)
        else:
            return 0.5
    
    def _stumpff_S(self, z):
        """Stumpff function S(z)"""
        if z > 0:
            sz = np.sqrt(z)
            return (sz - np.sin(sz)) / (z * sz)
        elif z < 0:
            sz = np.sqrt(-z)
            return (np.sinh(sz) - sz) / (-z * sz)
        else:
            return 1/6
    
    def find_transfer_orbits(self, a_init, e_init, r_target, v_target_dir, delta_v_max, 
                           num_nu=36, num_angles=12, dt_range=(0.2, 0.6)):
        """
        Find all valid initial orbits that can reach the target.
        """
        valid_transfers = []
        
        # Discretize true anomaly
        nu_values = np.linspace(0, 2*np.pi, num_nu)
        
        # Discretize other angles
        i_values = np.linspace(0, np.pi, num_angles)
        Omega_values = np.linspace(0, 2*np.pi, num_angles)
        omega_values = np.linspace(0, 2*np.pi, num_angles)
        
        # Search over all combinations
        for nu in nu_values:
            for i in i_values:
                for Omega in Omega_values:
                    for omega in omega_values:
                        # Get initial position and velocity
                        r0, v0 = self.orbital_elements_to_rv(a_init, e_init, i, Omega, omega, nu)
                        
                        # Try different transfer times (in normalized units)
                        for dt in np.linspace(dt_range[0], dt_range[1], 20):
                            # Solve Lambert's problem
                            v1, v2 = self.lambert_solver(r0, r_target, dt)
                            
                            if v1 is None:
                                continue
                            
                            # Check delta-v constraint
                            dv = norm(v1 - v0)
                            if dv > delta_v_max:
                                continue
                            
                            # Check arrival velocity direction
                            v2_dir = v2 / norm(v2)
                            if np.dot(v2_dir, v_target_dir) > 0.99:  # Within ~8 degrees
                                valid_transfers.append({
                                    'nu': nu, 'i': i, 'Omega': Omega, 'omega': omega,
                                    'dt': dt, 'dv': dv,
                                    'r0': r0, 'v0': v0, 'v1': v1, 'v2': v2,
                                    'arrival_speed': norm(v2)
                                })
        
        return valid_transfers
    
    def plot_transfer(self, transfer, fig=None):
        """
        Visualize a transfer orbit.
        """
        if fig is None:
            fig = plt.figure(figsize=(12, 10))
        
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot initial orbit
        nu_plot = np.linspace(0, 2*np.pi, 100)
        r_init_orbit = []
        for nu in nu_plot:
            r, _ = self.orbital_elements_to_rv(
                self.a_init, self.e_init, 
                transfer['i'], transfer['Omega'], transfer['omega'], nu
            )
            r_init_orbit.append(r)
        r_init_orbit = np.array(r_init_orbit)
        
        ax.plot(r_init_orbit[:, 0], r_init_orbit[:, 1], r_init_orbit[:, 2], 
                'b-', linewidth=2, label='Initial Orbit', alpha=0.7)
        
        # Plot transfer trajectory
        r0 = transfer['r0']
        rf = self.r_target
        dt = transfer['dt']
        
        # Propagate transfer orbit
        transfer_points = []
        for t in np.linspace(0, dt, 50):
            # This is simplified - for accurate visualization, integrate the orbit
            alpha = t / dt
            r = (1 - alpha) * r0 + alpha * rf  # Linear approximation
            transfer_points.append(r)
        transfer_points = np.array(transfer_points)
        
        ax.plot(transfer_points[:, 0], transfer_points[:, 1], transfer_points[:, 2], 
                'r--', linewidth=2, label='Transfer Trajectory')
        
        # Plot key points
        ax.scatter(*r0, color='green', s=100, label='Departure')
        ax.scatter(*rf, color='red', s=100, label='Target')
        
        # Plot velocity vectors
        scale = 0.2
        ax.quiver(*r0, *transfer['v0']*scale, color='green', alpha=0.5)
        ax.quiver(*r0, *transfer['v1']*scale, color='orange', alpha=0.8)
        ax.quiver(*rf, *transfer['v2']*scale, color='red', alpha=0.8)
        
        # Plot central body
        ax.scatter(0, 0, 0, color='yellow', s=200, label='Central Body')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'Transfer: Δv = {transfer["dv"]:.3f}, Δt = {transfer["dt"]:.3f}')
        
        return fig, ax

# Example usage
def main():
    """
    Example: GEO to LEO transfer with 45° inclination change
    
    This simulates a satellite transferring from geostationary orbit (GEO) 
    to low Earth orbit (LEO) with a plane change to 45° inclination.
    
    Units:
    - Distance: Earth radii (1 Earth radius = 6378 km)
    - Velocity: Normalized (1 unit ≈ 7.9 km/s at Earth surface)
    - Time: Normalized (1 unit ≈ 806.8 seconds)
    """
    # Create solver
    # Using normalized units: 1 DU = Earth radius (6378 km)
    # For Earth: mu = 398600.4 km³/s², so in normalized units:
    # mu = 398600.4 / (6378^3) * (6378)^3 / TU^3 where 1 TU = 806.8 s
    mu_earth = 1.0  # Normalized units
    solver = OrbitalTransferSolver(mu=mu_earth)
    
    # Define problem parameters for GEO to LEO transfer
    # GEO: altitude ~35,786 km = 5.6 Earth radii above surface = 6.6 Earth radii from center
    a_init = 6.6        # GEO semi-major axis in Earth radii
    e_init = 0.0        # GEO is circular
    
    # Target: LEO at 400 km altitude = 0.063 Earth radii above surface
    # Total radius = 1 + 0.063 = 1.063 Earth radii
    # Let's place the target on the equator for simplicity
    r_target_mag = 1.063
    r_target = np.array([r_target_mag, 0.0, 0.0])  # Target position on equator
    
    # Target velocity direction for 45° inclined orbit
    # At equator crossing, velocity is eastward but tilted 45° from equatorial plane
    # For circular LEO: v = sqrt(mu/r) = sqrt(1/1.063) ≈ 0.97 in normalized units
    v_leo = np.sqrt(mu_earth / r_target_mag)
    
    # Velocity direction: eastward (along y) but tilted 45° upward (z)
    v_target_dir = np.array([0.0, np.cos(np.pi/4), np.sin(np.pi/4)])  # 45° inclination
    v_target_dir = v_target_dir / norm(v_target_dir)  # Normalize
    
    # Maximum delta-v for GEO to LEO is typically ~1.5 km/s
    # In normalized units (v_circular at 1 Earth radius ≈ 7.9 km/s):
    # delta_v = 1.5 / 7.9 ≈ 0.19 in normalized units
    delta_v_max = 0.25   # Allowing some margin
    
    # Store parameters in solver for plotting
    solver.a_init = a_init
    solver.e_init = e_init
    solver.r_target = r_target
    
    print("Searching for valid GEO to LEO transfer orbits...")
    print(f"Initial orbit: GEO (a={a_init:.1f} Earth radii, circular)")
    print(f"Target: LEO at {(r_target_mag-1)*6378:.0f} km altitude, 45° inclination")
    print(f"Max Δv: {delta_v_max:.3f} normalized units ≈ {delta_v_max*7.9:.2f} km/s")
    print()
    
    valid_transfers = solver.find_transfer_orbits(
        a_init, e_init, r_target, v_target_dir, delta_v_max,
        num_nu=24, num_angles=8,  # Increased resolution for better coverage
        dt_range=(0.2, 0.6)  # 2.7 to 8 hours - appropriate for GEO to LEO
    )
    
    print(f"Found {len(valid_transfers)} valid transfers")
    
    if valid_transfers:
        # Sort by delta-v
        valid_transfers.sort(key=lambda x: x['dv'])
        
        # Plot the best few transfers
        fig = plt.figure(figsize=(15, 12))
        
        # 3D plot of best transfer
        ax1 = fig.add_subplot(221, projection='3d')
        best = valid_transfers[0]
        
        # Plot initial orbit
        nu_plot = np.linspace(0, 2*np.pi, 100)
        r_init_orbit = []
        for nu in nu_plot:
            r, _ = solver.orbital_elements_to_rv(
                a_init, e_init, best['i'], best['Omega'], best['omega'], nu
            )
            r_init_orbit.append(r)
        r_init_orbit = np.array(r_init_orbit)
        
        ax1.plot(r_init_orbit[:, 0], r_init_orbit[:, 1], r_init_orbit[:, 2], 
                'b-', linewidth=2, label='GEO', alpha=0.7)
        
        # Plot transfer
        ax1.plot([best['r0'][0], r_target[0]], 
                [best['r0'][1], r_target[1]], 
                [best['r0'][2], r_target[2]], 'r--', linewidth=2, label='Transfer')
        
        ax1.scatter(*best['r0'], color='green', s=100, label='Departure')
        ax1.scatter(*r_target, color='red', s=100, label='LEO Target')
        ax1.scatter(0, 0, 0, color='blue', s=300, label='Earth')
        
        # Add Earth sphere for reference
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, alpha=0.3, color='blue')
        
        ax1.set_xlabel('X (Earth radii)')
        ax1.set_ylabel('Y (Earth radii)')
        ax1.set_zlabel('Z (Earth radii)')
        ax1.legend()
        ax1.set_title(f'Best GEO→LEO Transfer: Δv = {best["dv"]:.3f} ≈ {best["dv"]*7.9:.2f} km/s')
        
        # Set equal aspect ratio and appropriate limits
        max_range = 7  # Slightly larger than GEO radius
        ax1.set_xlim([-max_range, max_range])
        ax1.set_ylim([-max_range, max_range])
        ax1.set_zlim([-max_range, max_range])
        
        # Delta-v distribution
        ax2 = fig.add_subplot(222)
        dvs = [t['dv'] for t in valid_transfers]
        dvs_kms = [dv * 7.9 for dv in dvs]  # Convert to km/s
        ax2.hist(dvs_kms, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Delta-v (km/s)')
        ax2.set_ylabel('Count')
        ax2.set_title('Delta-v Distribution')
        ax2.axvline(delta_v_max * 7.9, color='red', linestyle='--', label='Max Δv')
        ax2.legend()
        
        # Transfer time distribution
        ax3 = fig.add_subplot(223)
        dts = [t['dt'] for t in valid_transfers]
        dts_hours = [dt * 806.8 / 3600 for dt in dts]  # Convert to hours
        ax3.hist(dts_hours, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Transfer Time (hours)')
        ax3.set_ylabel('Count')
        ax3.set_title('Transfer Time Distribution')
        
        # True anomaly vs Delta-v
        ax4 = fig.add_subplot(224)
        nus = [t['nu'] * 180/np.pi for t in valid_transfers]
        dvs_kms_scatter = [t['dv'] * 7.9 for t in valid_transfers]
        ax4.scatter(nus, dvs_kms_scatter, alpha=0.6)
        ax4.set_xlabel('Departure True Anomaly (deg)')
        ax4.set_ylabel('Delta-v (km/s)')
        ax4.set_title('Departure Position vs Delta-v')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print best transfers
        print("\nBest 5 transfers:")
        print("Nu(deg)  i(deg)  Ω(deg)  ω(deg)  Δv(km/s)  Δt(min)  Arrival Speed")
        print("-" * 75)
        for t in valid_transfers[:5]:
            # Convert normalized units to physical units
            dv_kms = t['dv'] * 7.9  # km/s
            dt_min = t['dt'] * 806.8 / 60  # minutes (1 TU ≈ 806.8 seconds)
            v_arrival_kms = t['arrival_speed'] * 7.9  # km/s
            
            print(f"{t['nu']*180/np.pi:6.1f}  "
                  f"{t['i']*180/np.pi:6.1f}  "
                  f"{t['Omega']*180/np.pi:6.1f}  "
                  f"{t['omega']*180/np.pi:6.1f}  "
                  f"{dv_kms:8.3f}  "
                  f"{dt_min:7.1f}  "
                  f"{v_arrival_kms:6.3f}")
        
        print(f"\nNote: LEO circular velocity at {(r_target_mag-1)*6378:.0f} km is {v_leo*7.9:.2f} km/s")

if __name__ == "__main__":
    main()