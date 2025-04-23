rockets = {
    "Falcon 9 Full Thrust": {
        "stages": [
            {
                "stage_number": 1,
                "engines": [
                    {
                        "name": "Merlin 1D",
                        "count": 9,
                        "ISP_SL": 282,  # seconds
                        "ISP_vac": 311,  # seconds
                        "thrust_SL": 845,  # kN per engine
                        "thrust_vac": 981  # kN per engine
                    }
                ],
                "mass_empty": 25600,  # kg
                "mass_propellant": 395700  # kg
            },
            {
                "stage_number": 2,
                "engines": [
                    {
                        "name": "Merlin 1D Vacuum",
                        "count": 1,
                        "ISP_vac": 348,  # seconds
                        "thrust_vac": 934  # kN
                    }
                ],
                "mass_empty": 3900,  # kg
                "mass_propellant": 92670  # kg
            }
        ],
        "total_mass": 549000  # kg
    },
    "Atlas V": {
        "stages": [
            {
                "stage_number": 1,
                "engines": [
                    {
                        "name": "RD-180",
                        "count": 1,
                        "ISP_SL": 311.3,  # seconds
                        "ISP_vac": 337.8,  # seconds
                        "thrust_SL": 3827,  # kN
                        "thrust_vac": 4152  # kN
                    }
                ],
                "mass_empty": 21054,  # kg
                "mass_propellant": 284089  # kg
            },
            {
                "stage_number": 2,
                "engines": [
                    {
                        "name": "RL10A",
                        "count": 1,
                        "ISP_vac": 450.5,  # seconds
                        "thrust_vac": 99.2  # kN
                    }
                ],
                "mass_empty": 2316,  # kg
                "mass_propellant": 20830  # kg
            }
        ]
    },
    "Soyuz-2.1b": {
        "stages": [
            {
                "stage_number": 1,
                "engines": [
                    {
                        "name": "RD-107A",
                        "count": 4,
                        "ISP_SL": 262,  # seconds
                        "ISP_vac": 319,  # seconds
                        "thrust_SL": 838.5,  # kN per engine
                        "thrust_vac": 1021.3  # kN per engine
                    }
                ],
                "mass_empty": 15136,  # kg (total for 4 boosters)
                "mass_propellant": 156640  # kg (total for 4 boosters)
            },
            {
                "stage_number": 2,
                "engines": [
                    {
                        "name": "RD-108A",
                        "count": 1,
                        "ISP_SL": 255,  # seconds
                        "ISP_vac": 319,  # seconds
                        "thrust_SL": 792.5,  # kN
                        "thrust_vac": 990.2  # kN
                    }
                ],
                "mass_empty": 6545,  # kg
                "mass_propellant": 90100  # kg
            },
            {
                "stage_number": 3,
                "engines": [
                    {
                        "name": "RD-0124",
                        "count": 1,
                        "ISP_vac": 359,  # seconds
                        "thrust_vac": 297.9  # kN
                    }
                ],
                "mass_empty": 2355,  # kg
                "mass_propellant": 25400  # kg
            }
        ]
    }
}

thrusters = {
    "MR-104J": {
        "type": "monopropellant",
        "propellant": "hydrazine",
        "thrust": 527.0,  # N, midpoint of 440-614
        "ISP": 219.0,  # s, midpoint of 215-223
        "mass": 6.44  # kg
    },
    "MR-103G": {
        "type": "monopropellant",
        "propellant": "hydrazine",
        "thrust": 0.66,  # N, midpoint of 0.19-1.13
        "ISP": 213.0,  # s, midpoint of 202-224
        "mass": 0.33  # kg
    },
    "R-4D": {
        "type": "bipropellant",
        "propellants": "N2O4 / MMH",
        "thrust": 490.0,  # N
        "ISP": 312.0,  # s
        "mass": 3.63  # kg
    },
    "10N Bipropellant Thruster": {
        "type": "bipropellant",
        "propellants": "MMH / N2O4",
        "thrust": 9.25,  # N, midpoint of 6.0-12.5
        "ISP": 292.0,  # s
        "mass": 0.35  # kg
    },
    "SPT-100": {
        "type": "Hall thruster",
        "propellant": "xenon",
        "power": 1.35,  # kW
        "thrust": 0.083,  # N (83 mN)
        "ISP": 1604.0,  # s
        "mass": 4.0  # kg, approximate
    },
    "AEPS": {
        "type": "Hall thruster",
        "propellant": "xenon",
        "power": 12.5,  # kW
        "thrust": 0.6,  # N (600 mN)
        "ISP": 2800.0,  # s
        "mass": 47.0  # kg
    },
    "VACCO MiPS Cold Gas Thruster": {
        "type": "cold gas",
        "propellant": "R134a",
        "thrust": 0.01,  # N (10 mN)
        "ISP": 40.0  # s
    },
    "Busek Micro-Resistojet": {
        "type": "resistojet",
        "propellant": "gas",
        "thrust": 0.005,  # N, midpoint of up to 10 mN
        "ISP": 150.0  # s
    },
    "MR-502": {
        "type": "resistojet",
        "propellant": "hydrazine",
        "thrust": 0.8,  # N
        "ISP": 299.0,  # s
        "power": 885.0  # W
    },
    "Mira": {
        "type": "bipropellant",
        "propellants": "nitrous oxide / ethane",
        "thrust": 208.0,  # N, total from 8 Saiph thrusters (26 N each)
        "ISP": 290.0,  # s, per Saiph thruster
        "delta_v": 500.0,  # m/s for 300 kg payload (up to 900 m/s for 100 kg)
        "attitude_control": "4 reaction wheels, 16 cold-gas thrusters",
        "power": 500.0,  # W, max available
        "data_downlink": 4.0,  # Mbps, X-band
        "data_uplink": 0.4,  # Mbps, S-band
        "lifetime": 5.0,  # years, in GEO
        "redundancy": "single fault tolerant"
    }
}