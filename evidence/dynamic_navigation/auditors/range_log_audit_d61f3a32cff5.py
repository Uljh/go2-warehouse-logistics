"""Independent checks of recorded ranges delivered to the map/controller."""
from collections import Counter
import json
import math
from pathlib import Path


def audit_range_log(path, profile=None):
    """Inspect measurements as data, without calling the sensor implementation.

    A physical ray hitting the cart is distinct from a usable measured hit.
    The ideal profile's older logs omit the measurement block; in that case
    occupied endpoints and raw distance establish the usable return.
    """
    counts=Counter()
    raw_fixture=set()
    usable_fixture=set()
    scans=0
    contiguous=True
    valid_values=True
    invalid_clear=False
    endpoints_match=True
    coverage_ok=True
    minimum_coverage=1.
    max_quantization_error=0.
    native_endpoint_error=0.
    errors=[]
    previous=0
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            scans+=1
            contiguous &= row['scan']==previous+1
            previous=row['scan']
            layer_total=Counter()
            layer_valid=Counter()
            for beam in row['beams']:
                layer=beam['layer']
                layer_total[layer]+=1
                raw=beam.get('raw_distance_m')
                valid_values &= raw is None or (math.isfinite(raw) and 0.<=raw<=5.)
                cart='/AisleBlocker' in (beam.get('physical_hit_prim_path') or '')
                if cart and raw is not None:raw_fixture.add(row['scan'])
                measurement=beam.get('measurement')
                if measurement is None:
                    status='hit' if raw is not None else 'no_return'
                    distance=raw if raw is not None else 5.
                    valid_values &= profile is None
                else:
                    status=measurement['status']
                    distance=measurement['distance_m']
                counts[status]+=1
                occupied=beam.get('occupied_return')
                free=beam.get('free_segment')
                if status in ('dropout','out_of_range'):
                    invalid_clear |= occupied is not None or free is not None
                    valid_values &= distance is None
                    continue
                if status not in ('hit','no_return'):
                    valid_values=False
                    continue
                layer_valid[layer]+=1
                finite_distance=isinstance(distance,(int,float)) and math.isfinite(distance)
                valid_values &= finite_distance and 0.<=distance<=5.
                endpoints_match &= free is not None
                if not finite_distance:continue
                if free is not None:
                    endpoints_match &= len(free)==3 and free[2]==layer
                    endpoint_error=abs(math.dist(free[0],free[1])-distance)
                    if measurement is not None:
                        # Measured endpoints are explicitly projected from a
                        # scalar range. Older ideal logs instead store the
                        # native PhysX hit point, whose reported distance can
                        # differ slightly; do not reinterpret that API field.
                        endpoints_match &= endpoint_error<1.e-6
                    else:
                        native_endpoint_error=max(native_endpoint_error,endpoint_error)
                if status=='no_return':
                    endpoints_match &= occupied is None and distance==5. and raw is None
                    continue
                endpoints_match &= (raw is not None and occupied is not None and
                    len(occupied)==4 and occupied[2]==layer and
                    abs(occupied[3]-distance)<1.e-8 and free is not None and
                    math.dist(occupied[:2],free[1])<1.e-6)
                if cart and occupied is not None and raw is not None:
                    usable_fixture.add(row['scan'])
                if raw is not None:
                    errors.append(distance-raw)
                if profile is not None:
                    resolution=profile['resolution_m']
                    max_quantization_error=max(max_quantization_error,
                        abs(distance/resolution-round(distance/resolution))*resolution)
                    valid_values &= profile['minimum_range_m']<=distance<=profile['maximum_range_m']
            fractions=[layer_valid[k]/layer_total[k] for k in (0,1) if layer_total[k]]
            coverage_ok &= len(fractions)==2 and (profile is None or min(fractions)>=.8)
            minimum_coverage=min(minimum_coverage,min(fractions,default=0.))
    max_error=max(map(abs,errors),default=0.)
    mean_error=sum(errors)/len(errors) if errors else None
    bound=(3.*profile['noise_sigma_m']+profile['resolution_m']/2. if profile is not None else 0.)
    checks=dict(scans_present=scans>0,contiguous_scans=contiguous,
        finite_valid_measurements=valid_values,invalid_rays_never_clear_map=not invalid_clear,
        endpoints_use_measured_distances=endpoints_match,per_layer_coverage=coverage_ok,
        quantization=max_quantization_error<=1.e-8,
        bounded_hit_noise=max_error<=bound+1.e-7)
    return dict(passed=all(checks.values()),checks=checks,scans=scans,
        statuses=dict(counts),raw_fixture_scans=sorted(raw_fixture),
        usable_fixture_scans=sorted(usable_fixture),minimum_valid_layer_fraction=minimum_coverage,
        maximum_hit_error_m=max_error,hit_error_samples=len(errors),
        hit_error_mean_m=mean_error,
        hit_error_std_m=math.sqrt(sum((e-mean_error)**2 for e in errors)/len(errors)) if errors else None,
        maximum_quantization_error_m=max_quantization_error,
        maximum_native_endpoint_range_discrepancy_m=native_endpoint_error,
        calibrated_to_hardware=False)
