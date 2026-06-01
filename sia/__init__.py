"""SIA: Self-Improving AI framework"""

__version__ = "0.2.1"

# Spectral architecture (SIA²)
from sia.spectral_orchestrator import (
    SIA2Orchestrator,
    SpectralAnalyzer,
    ConservationChecker,
    ConservationLaw,
    BanachConvergence,
    InformationGeometry,
    PDEImprovementDynamics,
    RenormalizationTracker,
    ImprovementStep,
    ImprovementTrajectory,
    SpectralMode,
)

# Spectral integration layer
from sia.spectral_integration import (
    SpectralIntegration,
    extract_gen_metrics,
    inject_spectral_flag,
    run_with_spectral,
    wrap_orchestrator,
)
