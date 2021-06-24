
import mahotas 
from astropy.stats import median_absolute_deviation

@staticmethod
    def _extract_phenotype_extended(data_phenotype, nuclei, cells, wildcards, channel):
 
        def masked(region, index):
            return region.intensity_image_full[index][region.filled_image]

        def mahotas_zernike(region):
            mfeat = mahotas.features.zernike_moments(region.intensity_image_full[channel], radius = 9, degree=9)
            return mfeat

        def mahotas_pftas(region):
            mfeat = mahotas.features.pftas(region.intensity_image_full[channel])
            ### according to this, at least as good as haralick/zernike and much faster:
            ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
            return mfeat

        features_nuclear = {
            'dapi_nuclear_median': lambda r: np.median(masked(r, 0)),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'dapi_nuclear_int'   : lambda r: masked(r, 0).sum(),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'dapi_nuclear_max'   : lambda r: masked(r, 0).max(),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'area_nuclear'       : lambda r: r.area,
            'perimeter_nuclear' : lambda r: r.perimeter,
            'eccentricity_nuclear' : lambda r: r.eccentricity, #cell
            'major_axis_nuclear' : lambda r: r.major_axis_length, #cell
            'minor_axis_nuclear' : lambda r: r.minor_axis_length, #cell
            'orientation_nuclear' : lambda r: r.orientation,
            'zernike_features_nuclear': mahotas_zernike,
            'hu_moments_nuclear': lambda r: r.moments_hu,
            'solidity_nuclear': lambda r: r.solidity,
            'extent_nuclear': lambda r: r.extent,
            'pftas_features_nuclear': mahotas_pftas,
            'dapi_nuclear_sd': lambda r: np.std(masked(r,channel)),
            'dapi_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'dapi_nuclear_mean': lambda r: masked(r, 0).mean(),
            'dapi_nuclear_25': lambda r: np.percentile(masked(r, 0),25),
            'dapi_nuclear_75': lambda r: np.percentile(masked(r, 0),75),
            'channel_nuclear_min': lambda r: np.min(masked(r,channel)),
            'channel_nuclear_sd': lambda r: np.std(masked(r,channel)),
            'channel_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_nuclear_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_nuclear_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        }


        features_cell = {
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'dapi_cell_median'    : lambda r: np.median(masked(r, 0)),
            'area_cell'       : lambda r: r.area,
            'perimeter_cell' : lambda r: r.perimeter,
            'cell'            : lambda r: r.label,
            'euler_cell' : lambda r: r.euler_number,
            'eccentricity_cell' : lambda r: r.eccentricity, #cell
            'major_axis_cell' : lambda r: r.major_axis_length, #cell
            'minor_axis_cell' : lambda r: r.minor_axis_length, #cell
            'orientation_cell' : lambda r: r.orientation,
            'zernike_features_cell': mahotas_zernike,
            'hu_moments_cell': lambda r: r.moments_hu,
            'solidity_cell': lambda r: r.solidity,
            'extent_cell': lambda r: r.extent,
            'pftas_features_cell': mahotas_pftas,
            'channel_cell_min': lambda r: np.min(masked(r,channel)),
            'channel_cell_sd': lambda r: np.std(masked(r,channel)),
            'channel_cell_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cell_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cell_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        }


        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                .reset_index())

        return df
