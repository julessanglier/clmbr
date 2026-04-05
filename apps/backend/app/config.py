from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "clmbr"
    cors_origins: list[str] = ["*"]
    dem_dir: str = "dem_tiles"

    # Slope calculation
    sample_step_m: float = 15
    min_segment_length_m: float = 50
    max_realistic_slope: float = 18
    smoothing_sigma: float = 3
    min_elevation_change_m: float = 6
    min_slope_elevation_ratio: float = 0.7
    min_length_for_steep: float = 80

    class Config:
        env_prefix = "CLMBR_"


settings = Settings()
