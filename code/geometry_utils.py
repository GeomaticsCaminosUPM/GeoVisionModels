import geopandas as gpd
gpd.options.io_engine = "pyogrio"
#General functions
def string2file(string,filename):
    with open(filename,"w+") as f:
        f.writelines(string)

def gpd_any(x):
    try:
        for y in x.keys():
            if gpd_any(x[y]):
                return True
        return False
    except:
        return x == True
    
def gpd_all(x):
    try:
        for y in x.keys():
            if not gpd_all(x[y]):
                return False
        return True
    except:
        return x == True

def numpy_to_python(x):
    if "array" in str(type(x)):
        x = x.tolist()
    else:
        x = x.item()
    return x 

def check_x_y_points(x,y=None):
    if "numpy" in str(type(x)):
        x = numpy_to_python(x)

    if "numpy" in str(type(y)):
        y = numpy_to_python(y)

    if type(y) is not type(None):
        if type(x) is type(None):
            raise Exception("Give points as x=[[x1,y1],[x2,y2]] list or give x = [x1,x2] y = [y1,y2] kwargs")
        else:
            if type(x) != type(y):
                raise Exception(f"type x is {type(x)} but type y is different {type(y)}")
            if type(x) is not list and type(x) is not tuple:
                x = [x]
                y = [y]
            if len(x) != len(y):
                raise Exception(f"Length x is {len(x)} but len y is {len(y)} which is different")
            
            return x,y
            #points = []
            #for i in range(len(x)):
            #    points.append([x[i],y[i]])
            #return points
    else:
        if type(x) is list or type(x) is tuple:
            if type(x[0]) is not list and type(x[0]) is not tuple:
                x = [x]

            _x = []
            _y = []
            for i in x:
                _x.append(i[0])
                _y.append(i[1])
            
            return _x,_y
        else:
            raise Exception("Give points as x=[[x1,y1],[x2,y2]] list or give x = [x1,x2] y = [y1,y2] kwargs")
    
def get_asf_img_footprint(results,crs = 'EPSG:4326'):
    from shapely.geometry import shape
    import shapely.wkt
    import geopandas as gpd
    import asf_search as asf

    if type(results) is asf.ASFProduct:
        img_footprint = build_geometry(results, crs = 'EPSG:4326')
    elif type(results) is asf.ASFSearchResults or type(results) is list:
        image_footprints = []
        for res in results:
            image_footprints.append(shapely.wkt.loads(shape(res.geometry).wkt))
        img_footprint = build_geometries(image_footprints,crs='EPSG:4326')
    else:
        raise Exception("Error {} not implemented".format(type(results)))
    
    img_footprint = img_footprint.to_crs(crs)
    return img_footprint

# isce 

def dem_to_isce(dem,gdal_path = './usr/bin'):
    import os
    if not os.path.isfile(dem):
        raise Exception("{} file does not exist".format(dem))
    dem_isce = dem.split(".")[0] + ".dem.wgs84"
    try:
        os.system("gdal_translate -of GTiff {} {}".format(dem,dem_isce))
    except:
        os.system("{}/gdal_translate -of GTiff {} {}".format(gdal_path,dem,dem_isce))
    #!gdal_translate -of ISCE HERNANDO_2018_LiDAR_GCS_VCS_WGS84m.tif HERNANDO_2018_LiDAR_GCS_VCS_WGS84m.dem.wgs84
    try:
        import isce2
    except:
        import isce
    from applications.gdal2isce_xml import gdal2isce_xml
    xml_file = gdal2isce_xml(dem_isce)

    return dem_isce

# map functions 

def get_user_geom_from_map(m):
    import geopandas as gpd
    s = m.to_html()
    geoms = []
    idx = 0
    while True:
        start = s.find('"geometry":')
        if start == -1:
            break
        idx += 1
        end = s[start:].find("}")
        ss='{ "type": "Feature", '+s[start:start+end]+ '} }\n'
        geoms.append(ss)
        s = s[start+end:]
    geojson_str = '''{\n
        "type": "FeatureCollection",\n
        "features": [\n'''
    for i in geoms:
        geojson_str = geojson_str + i + ","
    geojson_str = geojson_str[0:-1]
    geojson_str = geojson_str + ']}'
    g=gpd.read_file(geojson_str,driver="GeoJSON")
    g.crs = 4326
    return build_geometry(g,crs=4326)


def to_basemap(m,name='Google Satellite Hybrid',transparent=False):
    import folium
    if type(name) is str:
        basemaps = {
            'Google Maps': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Maps',
                overlay = True,
                control = True,
                transparent = transparent
            ),
            'Google Satellite': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Satellite',
                overlay = True,
                control = True,
                transparent = transparent
            ),
            'Google Terrain': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Terrain',
                overlay = True,
                control = True,
                transparent = transparent
            ),
            'Google Satellite Hybrid': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Satellite',
                overlay = True,
                control = True,
                transparent = transparent
            ),
            'Esri Satellite': folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Esri Satellite',
                overlay = True,
                control = True,
                transparent = transparent
            )
        }
        basemaps[name].add_to(m)
    else:
        if "ipyleaflet.leaflet.WMSLayer" in str(type(name)):
            folium.raster_layers.WmsTileLayer(url = name.url,
                            layers = name.layers,
                            transparent = transparent, 
                            fmt="image/jpeg",
                            name = 'Background',
                            ).add_to(m)
        else:
            name.add_to(m)
    return None

#geometry helper functions

def buffer_in_m(geom,buffer):
    orig_crs = geom.crs
    utm = geom.to_crs(geom.estimate_utm_crs(datum_name='WGS 84'))
    utm = utm.buffer(buffer)
    return utm.to_crs(orig_crs)

def extract_linestrings_from_multilinestring(multilinestring):
    from shapely.geometry import MultiLineString, LineString
    if isinstance(multilinestring,LineString):
        return [multilinestring]
    
    if not isinstance(multilinestring, MultiLineString):
        raise ValueError(f"Input must be a MultiLineString geometry but got {type(multilinestring)}")

    linestrings = []
    for line in multilinestring.geoms:
        if isinstance(line, LineString):
            linestrings.append(line)
        else:
            raise ValueError(f"Invalid geometry type within MultiLineString {type(line)}")

    return linestrings

def force_2d_geom(geometry):
    import shapely
    import geopandas as gpd
    if type(geometry) is gpd.GeoDataFrame:
        crs = geometry.crs
        geometry.geometry = shapely.force_2d(geometry.geometry)
        geometry.crs = crs
    elif type(geometry) is gpd.GeoSeries:
        crs = geometry.crs
        geometry = shapely.force_2d(geometry)
        geometry.crs = crs
    else:
        geometry = shapely.force_2d(geometry)
       
    return geometry

def force_3d_geom(geometry):
    import shapely
    import geopandas as gpd
    if type(geometry) is gpd.GeoDataFrame:
        crs = geometry.crs
        geometry.geometry = shapely.force_3d(geometry.geometry)
        geometry.crs = crs
    elif type(geometry) is gpd.GeoSeries:
        crs = geometry.crs
        geometry = shapely.force_3d(geometry)
        geometry.crs = crs
    else:
        geometry = shapely.force_3d(geometry)
        
    return geometry

def check_z(geometry, has_z = None):
    if type(has_z) != type(None):
        if has_z:
            geometry = force_3d_geom(geometry)
        else:
            geometry = force_2d_geom(geometry)
            
    return geometry

def round_coordinates(geometry, decimals=2, has_z = None):
    from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
    from shapely.affinity import affine_transform
    import geopandas as gpd
    geometry = check_z(geometry,has_z=has_z)

    def round_coords(coords, decimals=2):
        if len(coords) == 2:
            return round(coords[0], decimals), round(coords[1], decimals)
        else:
            return round(coords[0], decimals), round(coords[1], decimals), round(coords[2], decimals)

    def round_geometry(geom, decimals):
        if geom.is_empty:
            return geom

        if geom.geom_type == 'Point':
            new_coords = round_coords(geom.coords[0], decimals)
            return Point(new_coords)

        elif geom.geom_type == 'LineString':
            new_coords = [round_coords(coord, decimals) for coord in geom.coords]
            return LineString(new_coords)

        elif geom.geom_type == 'Polygon':
            exterior_coords = [round_coords(coord, decimals) for coord in geom.exterior.coords]
            interior_coords = [
                [round_coords(coord, decimals) for coord in ring.coords] for ring in geom.interiors
            ]
            return Polygon(exterior_coords, interior_coords)

        elif geom.geom_type == 'MultiPoint':
            new_geoms = [round_geometry(point, decimals) for point in geom.geoms]
            return MultiPoint(new_geoms)

        elif geom.geom_type == 'MultiLineString':
            new_geoms = [round_geometry(line, decimals) for line in geom.geoms]
            return MultiLineString(new_geoms)

        elif geom.geom_type == 'MultiPolygon':
            new_geoms = [
                Polygon(
                    [round_coords(coord, decimals) for coord in poly.exterior.coords],
                    [
                        [round_coords(coord, decimals) for coord in ring.coords] for ring in poly.interiors
                    ],
                )
                for poly in geom.geoms
            ]
            return MultiPolygon(new_geoms)

        else:
            print(f"Warning. Round coordinates geom type {geom.geom_type} not implemented")
            # For other geometry types, you can add more cases as needed
            return geom
        
    if type(geometry) is gpd.GeoSeries or type(geometry) is gpd.GeoDataFrame:
        if len(geometry) == 0:
            return geometry 
        
        for i in range(len(geometry)):
            geometry.geometry[i] = round_geometry(geometry.geometry[i],decimals)

        return geometry
    elif type(geometry) is list:
        for i in range(len(geometry)):
            geometry[i] = round_geometry(geometry[i], decimals)

        return geometry
    else:
        return round_geometry(geometry, decimals)

def geometry_from_wkt(wkt,crs=4326):
    import geopandas as gpd
    from shapely import from_wkt
    x = gpd.GeoSeries(from_wkt(wkt),crs = crs)
    return x
    
def geoseries_from_bounds(bounds,crs=4326):
    from shapely.geometry import box
    import geopandas as gpd
    min_lon = bounds[0]
    min_lat = bounds[1]
    max_lon = bounds[2]
    max_lat = bounds[3]
    if max_lat < min_lat:
        min_lat,max_lat = max_lat,min_lat
    if max_lon < min_lon:
        min_lon,max_lon = max_lon,min_lon
    bbox = tuple([min_lon,min_lat,max_lon,max_lat])
    bbox = tuple(bbox)
    geom = gpd.GeoSeries([box(*bbox)],crs=crs) # xmin, ymin, xmax, ymax
    return geom

def array_to_polygon(x):
    from shapely.geometry import Polygon
    return Polygon(x)
def array_to_linestring(x):
    from shapely.geometry import LineString
    return LineString(x)
def array_to_point(x):
    from shapely.geometry import Point
    return Point(x)
def geoseries_from_shapely(x,crs=4326):
    import geopandas as gpd
    return gpd.GeoSeries(x,crs = crs)

#def geoseries_from_str(x,crs=4326):
#    from shapely.geometry import shape
#    import geopandas as gpd
#    return gpd.GeoSeries(shape(x),crs = crs)    

def concat_geoseries(geoseries:list,crs=None):
    import pandas as pd
    import geopandas as gpd

    if type(crs) == type(None):
        crs = geoseries[0].crs

    geodataframe = False
    for i in range(len(geoseries)):
        if "GeoDataFrame" in str(type(geoseries[i])):
            geodataframe = True 

        geoseries[i] = geoseries[i].to_crs(crs)

    c = pd.concat(geoseries)
    c = c.reset_index(drop=True)
    if not geodataframe:
        c = c.geometry

    return c

def merge_geoseries(x,merge="union",crs=None):
    import shapely.wkt
    import geopandas as gpd
    if "GeoDataFrame" in str(type(x)):
        x = x.geometry 

    if type(crs) is type(None):
        crs = x.crs 
        if type(x.crs) is type(None):
            raise Exception("crs not defined")
        
    if type(x.crs) is type(None):
        x.crs = crs
    else:
        x = x.to_crs(crs)

    g = []
    for i in x.to_wkt():
        g.append(shapely.wkt.loads(i))

    if merge == 'intersect' or merge == 'intersection' or merge == 'intersects' or merge == 'intersect-union' or merge == 'intersection-union':
        from shapely import intersection_all
        g = build_geometry(intersection_all(g), crs = crs)
        if type(g) is not type(None):
            return g
        elif merge == 'intersect-union' or merge == 'intersection-union':
            merge = "union"
        else:
            return gpd.GeoSeries([],crs=crs) 
        
    if merge == 'union' or merge == 'merge' or merge == 'add' or merge == 'unite':
        from shapely.ops import unary_union
        g = build_geometry(unary_union(g), crs = crs)
        return g.make_valid()
    else:
        raise Exception(f"merge method ({merge}) not implemented")

def geoseries_crop(G,g,both_geoms:bool=False):
    import geopandas as gpd
    if "GeoDataFrame" in str(type(G)):
        G_is_geodataframe = True 
    else:
        G_is_geodataframe = False
        G = gpd.GeoDataFrame(geometry=G)

    if "GeoDataFrame" in str(type(g)):
        g_is_geodataframe = True 
    else:
        g_is_geodataframe = False 
        g = gpd.GeoDataFrame(geometry=g)  

    g_crs = g.crs
    g = g.to_crs(G.crs) 

    resG = gpd.sjoin(G, g, how='inner', predicate='intersects')
    resG = resG.reset_index(drop=True)
    if G_is_geodataframe:
        resG = resG.drop(columns = ['index_right'])

    if both_geoms:
        resg = gpd.sjoin(g, G, how='inner', predicate='intersects')
        if not G_is_geodataframe:
            resG = resG.geometry

        if g_is_geodataframe:
            resg = resg.drop(columns = ['index_left'])
        else:
            resg = resg.geometry

        resG = resG[resG.is_valid].reset_index(drop=True)
        resg = resg[resg.is_valid].reset_index(drop=True)
        resg = resg.to_crs(g_crs)
        return resG, resg
    else:
        if not G_is_geodataframe:
            resG = resG.geometry

        resG = resG[resG.is_valid].reset_index(drop=True)
        return resG

def intersect_geoseries(G,g):
    import geopandas as gpd
    import shapely, copy
    if "GeoDataFrame" in str(type(G)):
        G_is_geodataframe = True 
    else:
        G_is_geodataframe = False

    if "GeoDataFrame" in str(type(g)):
        g = g.geometry 

    g = g.to_crs(G.crs)
    G,g = geoseries_crop(G,g,both_geoms=True)
    if len(g) == 0 or len(G) == 0:
        if G_is_geodataframe:
            return gpd.GeoDataFrame([],geometry=[],crs=G.crs)
        else:
            return gpd.GeoSeries([],crs=G.crs)
    
    if G_is_geodataframe:
        res = copy.deepcopy(G)
        res.geometry = G.geometry.intersection(merge_geoseries(g,merge='union')[0])
        res.geometry = res.geometry[res.geometry.is_empty == False].make_valid().reset_index(drop=True)
    else:
        res = G.intersection(merge_geoseries(g,merge='union')[0])
        res = res[res.is_empty == False].make_valid().reset_index(drop=True)

    res = res[res.is_valid].reset_index(drop=True)

    if len(res) == 0:
        if G_is_geodataframe:
            return gpd.GeoDataFrame([],geometry=[],crs=G.crs)
        else:
            return gpd.GeoSeries([],crs=G.crs)
    else:
        return res


# build, load and save geometry functions

def read_file_to_geoseries(file,crs=None): #inacabado
    import os
    import geopandas as gpd
    gpd.options.io_engine = "pyogrio"
    if not os.path.isfile(file):
        raise Exception(f"file {file} not found")
    
    x = gpd.read_file(os.path.normpath(file))
    if type(x.crs) is type(None):
        x.crs = crs 

    elif type(crs) is not type(None):
        x = x.to_crs(crs)

    if type(x.crs) is type(None):
        raise Exception("crs not defined")    
    
    return x.geometry

def read_file_to_geodataframe(file,crs=None):
    import os
    import geopandas as gpd
    gpd.options.io_engine = "pyogrio"
    if not os.path.isfile(file):
        raise Exception(f"file {file} not found")
    
    x = gpd.read_file(os.path.normpath(file))

    if type(x.crs) is type(None):
        x.crs = crs

    elif type(crs) is not type(None):
        x = x.to_crs(crs)

    if type(x.crs) is type(None):
        raise Exception("crs not defined")
    
    return x   

def save_geoseries(geoseries,name:str,overwrite:bool=False,driver="GeoJSON"):
    import os
    #if ".geojson" not in name and "." not  in name:
    #    name = name + ".geojson"
    #elif ".geojson" not in name:
    #    raise Exception(f"only .geojson format accepted. {name}")
    if os.path.isfile(name):
        if overwrite == False:
            raise Exception(f"File {name} already exists. Set overwrite to True to overwrite")
        else:
            print(f"Overwriting file {name}")

    geoseries.to_file(name,driver = driver)
    #print(f"File {name} saved")

def build_geometries(x,crs = None,y=None, has_z = None):
    import os
    import geopandas as gpd
    import numpy as np
    import shapely 

    if "numpy" in str(type(x)):
        x = numpy_to_python(x)
    
    if "Collection" in str(type(x)) or "Multi" in str(type(x)):
        x = shapely.get_parts(x)

    if type(x) == type(None):
        if type(crs) == type(None):
            crs = 4326 

        return gpd.GeoSeries([],crs=crs) #None

    if type(crs) is list or type(crs) is tuple:
        geom_crs, crs = crs, crs[0]
    else:
        geom_crs, crs = [crs], crs

    if type(x) is str:
        if os.path.isfile(x):
            x = read_file_to_geoseries(x,crs=crs)
        else:
            x = build_geometry(x,crs=crs)
    
    elif type(y) != type(None):
        x,y = check_x_y_points(x=x,y=y)
        x = gpd.GeoSeries.from_xy(x,y,crs=crs)

    elif type(x) is list or type(x) is tuple:
        if len(x) > 0:
            x0 = x[0]
        else:
            x0 = None

        if "numpy" in str(type(x0)):
            x0 = numpy_to_python(x0)

        if type(x0) is int or type(x0) is float:
            x = build_geometries(x,crs=crs)
        else:
            if len(geom_crs) < len(x):
                for i in range(len(x) - len(geom_crs)):
                    geom_crs.append(geom_crs[-1])

            G = []
            for i in range(len(x)):
                if type(x[i]) is str and os.path.isfile(x[i]):
                    g = read_file_to_geoseries(x[i],crs=geom_crs[i])
                elif type(x[i]) is gpd.GeoSeries or type(x[i]) is gpd.GeoDataFrame:
                    if type(x[i]) is gpd.GeoDataFrame:
                        g = x[i].geometry
                    else:
                        g = x[i] 

                else:
                    g = build_geometry(x[i],crs=geom_crs[i])
                    if len(g) == 0:
                        continue

                if type(crs) == type(None):
                    crs = g.crs
                else:
                    g = g.to_crs(crs)

                G.append(g)
            if len(G) == 0:
                if type(crs) == type(None):
                    crs = 4326 

                return gpd.GeoSeries([],crs=crs) 
            
            #x = gpd.GeoSeries([*G],crs=crs)
            x = concat_geoseries(G,crs=crs)

    if type(x) is gpd.GeoDataFrame:
        x = x.geometry
    
    x=x.reset_index(drop=True).geometry
    if type(x) is gpd.GeoSeries:
        x = x[x.is_valid]
        x = x[x != None]
        x = x[x.is_empty == False]
        x = x[x.isna() == False]
        if len(x) == 0:
            if type(x.crs) != type(None):
                crs = x.crs

            if type(crs) == type(None):
                crs = 4326 
                
            return gpd.GeoSeries([],crs=crs)  
        else:
            if type(crs) is not type(None):
                x = x.to_crs(crs)

            x = x[x.is_valid].reset_index(drop=True)
            return check_z(x,has_z=has_z)
    else:
        raise Exception(f"geometries type {type(x)} not implemented")


def build_geometry(x,buffer = 0,crs = None, merge = 'union',y=None,has_z = None):
    import geopandas as gpd
    import os, warnings, copy
    from shapely.geometry import shape
    from pyproj import CRS

    try:
        import asf_search as asf
        is_asf = True
    except:
        is_asf = False

    if type(x) == type(None):
        if type(crs) == type(None):
            crs = 4326 
                    
        return gpd.GeoSeries([],crs=crs)
        
    if "numpy" in str(type(x)):
        x = numpy_to_python(x)

    t = type(x)
    if t is str:
        if os.path.isfile(x):
            x = read_file_to_geoseries(x,crs=crs)
        else:
            if crs == None:
                crs = 4326

            try:
                x = shape(x)
            except:
                try:
                    x = geometry_from_wkt(x,crs=crs)
                except:
                    if type(crs) == type(None):
                        crs = 4326 
                    
                    warnings.warn("Can't read {}. String can not be read as file or geometry. Returning empty".format(x))
                    return gpd.GeoSeries([],crs=crs) 
                
    elif t is int or t is float:
        if type(y) != type(None):
            x,y = check_x_y_points(x=x,y=y)
            x = gpd.GeoSeries.from_xy(x,y)

    crs_is_none = False
    if type(crs) is type(None):
        crs = 4326
        crs_is_none = True 
    
    crs = CRS.from_user_input(crs)

    t = type(x)    
    if t is list or t is tuple:
        if len(x) == 0:
            if type(crs) == type(None):
                crs = 4326 

            return gpd.GeoSeries([],crs=crs) 
    
        if len(x) > 0:
            x0 = x[0]
        else:
            x0 = None

        if "numpy" in str(type(x0)):
            x0 = numpy_to_python(x0)

        if type(y) == type(None) and len(x) == 4 and (type(x0) is int or type(x0) is float):
            x = geoseries_from_bounds(x,crs=crs)
        else:
            x = check_x_y_points(x=x,y=y) 
            if type(x0) is list or type(x0) is tuple:
                if len(x0) == 2:
                    if len(x) == 1:
                        x = array_to_point(x)
                    elif abs(x[0][0] - x[-1][0]) < 0.01 and abs(x[0][1] - x[-1][1]) < 0.01:
                        x = array_to_polygon(x)
                    else:
                        x = array_to_linestring(x)

                else:
                    raise Exception(f"Could not interpret array {x} with datatype {type(x)} as a valid geometry")
                
            else:
                raise Exception(f"Could not interpret array {x} with datatype {type(x)} as a valid geometry")
            
    t = type(x)
    if "shapely" in str(t):
        x = geoseries_from_shapely(x,crs=crs)
    elif t is gpd.GeoDataFrame:
        x = x.geometry
    elif is_asf and (t is asf.ASFProduct or t is asf.ASFSearchResults):
        if t is asf.ASFProduct:
            x = gpd.GeoSeries(shape(x[0].geometry).wkt, crs = 4326)
            x = x.to_crs(crs)
        elif t is asf.ASFSearchResults:
            x = get_asf_img_footprint(x,crs = 4326)

    t = type(x)
    if t is gpd.GeoSeries:
        #x = x[(x.isna() + x.is_empty + [i == None for i in x]) == 0]
        x = x[x.is_valid]
        x = x[x != None]
        x = x[x.is_empty == False]
        x = x[x.isna() == False]

        if len(x) == 0:
            if type(x.crs) != type(None):
                crs = x.crs

            if type(crs) == type(None):
                    crs = 4326 

            return gpd.GeoSeries([],crs=crs) 
        elif len(x) > 1:
            x = merge_geoseries(x,merge=merge,crs=x.crs)

        if type(x.crs) is type(None):
            x.crs = crs

        if crs_is_none:
            crs = x.crs
        else:
            x = x.to_crs(crs)

    else:
        raise Exception(f"geometry type {t} not implemented")
    
    if buffer > 0:
        local_crs = x.estimate_utm_crs(datum_name='WGS 84')
        crs = copy.copy(x.crs)
        x = x.to_crs(local_crs)
        x = x.buffer(buffer)
        x = x.to_crs(crs)
    
    x = x[x.is_valid].reset_index(drop=True)
    return check_z(x,has_z)

# geometry bounds functions

def get_bounds(geom,crs=None):
    print("Warning. geometry_utils.get_bounds() depreceated")
    geom = build_geometry(geom,crs=crs)
    bounds = geom.total_bounds
    return bounds
    
def get_bounds_of_sentinel_zip(file):
    import zipfile, os
    with zipfile.ZipFile(file) as zf:
        l = zf.namelist()
        kml = None 
        tif = None
        for s in l:
            if kml == None and s.find(".kml") != -1:
                kml = s
            elif tif == None and s.find(".tif") != -1 or s.find(".tiff") != -1:
                tif = s 
        if kml != None:
            with zf.open(kml, mode="r") as _file:
                import io
                lines = []
                for l in io.TextIOWrapper(_file, encoding="utf-8"):
                    lines.append(l)
                return get_bounds_of_kml(lines) 
        elif tif != None:
            with zf.open(tif) as _file:
                path = os.path.dirname(file) 
                _file.extract("/temp_tif.tif", path = path)
                bounds = get_bounds_of_img(file) 
                os.remove(path + "/temp_tif.tif")
                return bounds       
        else:
            raise Exception("No .kml or .tif .tiff file found in .zip file: ",file)

def get_bounds_of_kml(file):
    if type(file) is str:
        file = open(file, 'r').readlines()
    if type(file) is list or type(file) is tuple:
        for f in file:
            if f.find("<coordinates>") != -1:
                init = f.find("<coordinates>") + len("<coordinates>")
                end = f.find("</coordinates>") - 1
                c = f[init:end]
                c = c.split(" ")
                pol = []
                for p in c:
                    pp = p.split(",")
                    pol.append([float(pp[0]),float(pp[1])])
                return build_geometry(pol)
    else:
        raise Exception("Error type(file) not implemented: ",type(file))

def get_crs_of_img(file):
    if type(file) is str:
        import rasterio as rio
        src = rio.open(file)
    else:
        src = file
    
    crs = src.crs.to_proj4()    
    src.close()
    return crs

def get_bounds_of_img(file,crs=None):
    from pyproj import Transformer
    from pyproj import CRS
    
    if type(file) is str:
        import rasterio as rio
        src = rio.open(file)
    else:
        src = file
    
    if type(crs) != type(None):
        crs = CRS.from_user_input(crs)
    else:
        crs = src.crs.to_proj4()

    transformer = Transformer.from_crs(src.crs.to_proj4(), src.crs.to_proj4()) #De lo que sea a geograficas WGS84
    src.close()
    r = transformer.transform_bounds(src.bounds.left,src.bounds.bottom,src.bounds.right, src.bounds.top)
    r = build_geometry([r[0],r[1],r[2],r[3]],crs=src.crs.to_proj4()).to_crs(crs)
    return r


# dates functions

def get_sentinel_img_date(img,dateinds=None,dateformat='%Y%m%dT%H%M%S',get_all_dates=False):
    import os, re, datetime
    img = os.path.split(img)[-1]
    if type(dateformat) == type(None):
        dateformat = '%Y%m%dT%H%M%S'
    if type(dateinds) == type(None):
        d = re.findall("\d\d\d\d\d\d\d\dT\d\d\d\d\d\d",img)
    else:
        if len(dateinds) == 2 and dateinds[0] < dateinds[1]:
            d = [img[dateinds[0]:dateinds[1]]]
        else:
            raise Exception("dateinds = {} should be a list [min,max] pointing to the part of the filename containing the image date.".format(dateinds))
    if d == []:
        raise Exception("Date of image {} not found.".format(img))
    
    dates = []
    if get_all_dates:
        for i in d:
            dates.append(datetime.datetime.strptime(i,dateformat))
    else:
        dates = datetime.datetime.strptime(d[0],dateformat)
    return dates

def str_to_datetime(date):
    import datetime
    if type(date) is not str:
        return date
    
    continue_try = False
    try:
        d = datetime.datetime.strptime(date,'%Y-%m-%dT%H:%M:%S.%fZ')
    except:
        try:
            d = datetime.datetime.strptime(date,'%Y-%m-%dT%H:%M:%S.%f')
        except:
            try:
                d = datetime.datetime.strptime(date,'%Y-%m-%dT%H:%M:%SZ')
            except:
                try:
                    d = datetime.datetime.strptime(date,'%Y-%m-%dT%H:%M:%S')
                except:
                    try:
                        d = datetime.datetime.strptime(date,'%Y-%m-%d')
                    except:
                        try:
                            d = datetime.datetime.strptime(date,'%Y-%m')
                        except:
                            try:
                                d = datetime.datetime.strptime(date,'%Y')
                            except:
                                continue_try = True
    if continue_try:
        try:
            d = datetime.datetime.strptime(date,'%Y%m%dT%H%M%S.%fZ')
        except:
            try:
                d = datetime.datetime.strptime(date,'%Y%m%dT%H%M%S.%f')
            except:
                try:
                    d = datetime.datetime.strptime(date,'%Y%m%dT%H%M%SZ')
                except:
                    try:
                        d = datetime.datetime.strptime(date,'%Y%m%dT%H%M%S')
                    except:
                        try:
                            d = datetime.datetime.strptime(date,'%Y%m%d')
                        except:
                            raise Exception("Unable to parse {} as date. Expected format is %Y-%m-".format(date)+"%"+"dT%H:%M:%S or %Y%m"+"%"+"dT%H%M%S")
    return d


def datetime_to_str(date):
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


def check_isolated_pol(pols,i,prefer_small_pols,min_intersection_length):
    inter = pols[i].buffer(0.01).intersection(pols)
    inter = inter[inter.is_empty == False]
    near_geoms = inter[(inter.type == "Polygon") + (inter.type == "MultiPolygon") + (inter.type == "LineString") + (inter.type == "MultiLineString")]
    near_geoms = near_geoms[near_geoms.length > min_intersection_length]
    if prefer_small_pols:
        add_inds = list(near_geoms.index[pols[near_geoms.index].area < pols[i].area])
    else:
        add_inds = list(near_geoms.index[pols[near_geoms.index].area > pols[i].area])

    visited_inds = []
    if len(add_inds) > 0:
        visited_inds.append(i)
        inds = []
        for j in add_inds:
            k, new_visited = check_isolated_pol(pols,j,prefer_small_pols,min_intersection_length)
            visited_inds += new_visited
            if j in k:
                inds.append(j)
            else:
                visited_inds.append(j)

        if len(inds) == 0:
            return [i], visited_inds
        else:
            return inds, visited_inds
    else:
        return [i], visited_inds
    
def create_isolated_pols(pols,prefer_small_pols = True,min_intersection_length=0):
    import numpy as np
    while True:
        inds = []
        visited_inds = []
        for i in pols.index:
            if i in visited_inds: 
                continue

            near_inds, new_visited_inds = check_isolated_pol(pols,i,prefer_small_pols,min_intersection_length)
            visited_inds += new_visited_inds
            visited_inds = list(np.unique(visited_inds))
            if len(near_inds) == 1 and near_inds[0] == i:
                inds.append(i)

        inds = np.unique(inds)
        if len(inds) == len(pols):
            return pols
        else:
            pols = pols[inds].reset_index(drop=True)
        
    
def linestrings_to_polygons(target_linestrings, other_linestrings=None, bounds=None, buffer=None,
                             snap_dist_far=0, snap_dist_close=0.1, grid_size=None, 
                             min_target_len=0, max_target_len=0, min_pol_area = 0, max_pol_area = 0, min_edges = 4, max_edges = 0,
                             decimals=3, prefer_small_pols=True, isolated_pols=True, allow_overlap=False, has_z = False):
    
    import shapely, copy
    from shapely.geometry import MultiLineString, LineString, Point
    import geopandas as gpd
    from datetime import datetime

    if type(bounds) != type(None) and type(buffer) == type(None):
        buffer = 20
    

    if type(grid_size) == type(None):
        grid_size = snap_dist_close / 20 

    
    target_linestrings = build_geometries(target_linestrings,has_z=has_z)
    orig_crs = target_linestrings.crs

    if type(bounds) == type(None):
        utm_crs = target_linestrings.estimate_utm_crs(datum_name='WGS 84')
    else:
        bounds = build_geometry(bounds,crs=orig_crs)
        utm_crs = bounds.estimate_utm_crs(datum_name='WGS 84')

    target_linestrings = target_linestrings.to_crs(utm_crs)

    if type(other_linestrings) != type(None):
        other_linestrings = build_geometries(other_linestrings,has_z=has_z)
        if len(other_linestrings) == 0:
            other_linestrings = None 
        else:
            other_linestrings = other_linestrings.to_crs(utm_crs)

    if type(bounds) != type(None):
        bounds = build_geometry(bounds,has_z=has_z)
        if len(bounds) == 0 or gpd_all(bounds.is_empty):
            print("bounds geometry is wrong. Setting bounds to None.")
            bounds = None
        else:
            bounds = bounds.to_crs(utm_crs)
            bounds_orig = copy.copy(bounds)
            bounds = bounds.simplify(snap_dist_close)

    if type(bounds) != type(None):

        if type(buffer) != type(None):
            bounds = bounds.buffer(buffer,cap_style='square',join_style='mitre') #####################################

        target_linestrings = intersect_geoseries(target_linestrings,bounds)
        if gpd_all(target_linestrings.is_empty):
            return gpd.GeoSeries([],crs=orig_crs)
        

        if type(other_linestrings) != type(None):
            other_linestrings = intersect_geoseries(other_linestrings,bounds)


        if type(buffer) != type(None):
            bounds = geoseries_crop(bounds,target_linestrings.buffer(buffer,cap_style='square',join_style='mitre'))

        bounds = bounds.geometry[0]
        if "Polygon" in str(type(bounds)):
            bounds = bounds.boundary

    target_linestrings[target_linestrings.type == "Polygon"] = target_linestrings[target_linestrings.type == "Polygon"].buffer(0.000001,cap_style='square',join_style='mitre').boundary
    target_linestrings[target_linestrings.type == "MultiPolygon"] = target_linestrings[target_linestrings.type == "MultiPolygon"].buffer(0.000001,cap_style='square',join_style='mitre').boundary
    target_linestrings = round_coordinates(target_linestrings,decimals,has_z)
    target_linestrings = target_linestrings.simplify(snap_dist_close)

    if type(other_linestrings) is type(None):
        other_linestrings = []
        target_linestrings = shapely.unary_union(target_linestrings)
        target_linestrings = target_linestrings.simplify(snap_dist_close)
        target_linestrings = build_geometries(shapely.get_parts(target_linestrings),crs=utm_crs)
        target_linestrings = target_linestrings[target_linestrings.length > grid_size].reset_index(drop=True)
    else:
        other_linestrings[other_linestrings.type == "Polygon"] = other_linestrings[other_linestrings.type == "Polygon"].buffer(0.000001,cap_style='square',join_style='mitre').boundary
        other_linestrings[other_linestrings.type == "MultiPolygon"] = other_linestrings[other_linestrings.type == "MultiPolygon"].buffer(0.000001,cap_style='square',join_style='mitre').boundary
        other_linestrings = round_coordinates(other_linestrings,decimals,has_z)
        other_linestrings = other_linestrings.simplify(snap_dist_close)

        target_linestrings = build_geometries(shapely.get_parts(target_linestrings),crs=utm_crs)
        if type(buffer) != type(None):
            other_linestrings = geoseries_crop(other_linestrings,target_linestrings.buffer(buffer,cap_style='square',join_style='mitre'))

        other_linestrings = shapely.unary_union(other_linestrings)
        target_linestrings = shapely.unary_union(target_linestrings)

        inter = build_geometries(
            shapely.get_parts(shapely.intersection(target_linestrings.buffer(snap_dist_close/2),other_linestrings)),crs=utm_crs
        )
        if type(inter) != type(None):
            inter = inter[inter.isna() == False]
            inter = inter[inter.is_empty == False]
            if len(inter) > 0:
                inter = list(inter.centroid.geometry)
                inter = shapely.geometry.MultiPoint(inter)
                target_linestrings = shapely.ops.split(target_linestrings,inter.buffer(grid_size/5))
                other_linestrings = shapely.ops.split(other_linestrings,inter.buffer(grid_size/5))   

        target_linestrings = shapely.unary_union(target_linestrings,grid_size=grid_size)
        target_linestrings = target_linestrings.simplify(snap_dist_close)
        target_linestrings = build_geometries(shapely.get_parts(target_linestrings),crs=utm_crs)
        target_linestrings = target_linestrings[target_linestrings.length > snap_dist_close/1.25].reset_index(drop=True)

        other_linestrings = shapely.unary_union(other_linestrings,grid_size=grid_size)
        other_linestrings = other_linestrings.simplify(snap_dist_close)             
        other_linestrings = shapely.get_parts(other_linestrings)
        other_linestrings = build_geometries(shapely.get_parts(other_linestrings),crs=utm_crs)
        other_linestrings = other_linestrings[other_linestrings.length > snap_dist_close/1.25].reset_index(drop=True)

    target_geom = shapely.unary_union(target_linestrings,grid_size=grid_size)
    if type(bounds) == type(None):
        union = shapely.unary_union([*target_linestrings,*other_linestrings])
    else:
        union = shapely.unary_union([*target_linestrings,*other_linestrings,bounds])

    if "Multi" not in str(type(union)):
        union = MultiLineString(shapely.get_parts(union))
    
    union = union.simplify(snap_dist_close)
    union = shapely.unary_union(union,grid_size=grid_size) 

    union = [o for o in union.geoms]

    for i in range(len(union)):
        helper = MultiLineString([*union[0:i],*union[i+1:]])
        union[i] = shapely.snap(union[i],helper,snap_dist_close)
        p0 = Point(union[i].coords[0])
        p1 = Point(union[i].coords[-1])
        if shapely.intersects(p0.buffer(grid_size/2),helper) == False:
            snap_p0 = shapely.snap(p0,helper,snap_dist_far) 
        else:
            snap_p0 = p0

        if shapely.intersects(p1.buffer(grid_size/2),helper) == False:
            snap_p1 = shapely.snap(p1,helper,snap_dist_far) 
        else:
            snap_p1 = p1

        union[i] = LineString(list(snap_p0.coords) + list(union[i].coords) + list(snap_p1.coords))
        union[i] =  union[i].simplify(snap_dist_close)

    union = shapely.unary_union(union)
    pols = build_geometries(shapely.get_parts(shapely.ops.polygonize(union)),crs=utm_crs)
    pols = pols.buffer(-grid_size,cap_style='square',join_style='mitre')
    pols = pols[pols.is_valid].reset_index(drop=True)
    pols = pols.buffer(grid_size,cap_style='square',join_style='mitre')
    pols = pols[pols.is_valid].reset_index(drop=True)
    pols = pols.simplify(snap_dist_close)
    pols = round_coordinates(pols,decimals,has_z)

    if max_pol_area > 0: 
        pols = pols[pols.area <= max_pol_area].reset_index(drop=True)

    if min_pol_area > 0: 
        pols = pols[pols.area >= min_pol_area].reset_index(drop=True)

    if min_edges > 0:
        pols = pols[pols.get_coordinates().index.value_counts().sort_index() >= min_edges].reset_index(drop=True)

    if max_edges > 0:
        pols = pols[pols.get_coordinates().index.value_counts().sort_index() <= max_edges].reset_index(drop=True)

    if len(pols) == 0:
        return gpd.GeoSeries([],crs=orig_crs)

    if not allow_overlap:
        pols = pols[pols.buffer(-snap_dist_close/4,cap_style='square',join_style='mitre').intersects(merge_geoseries(pols.boundary)[0]) == False].reset_index(drop=True)
        #pols = pols[pols.buffer(-snap_dist_close/2).intersection(union).is_empty].reset_index(drop=True)
        if len(pols) == 0:
            return gpd.GeoSeries([],crs=orig_crs)

    if min_target_len > 0:
        pols = pols[pols.boundary.length >= min_target_len * 2.1].reset_index(drop=True) #################################################

    if len(pols) == 0:
        return gpd.GeoSeries([],crs=orig_crs)

    if type(other_linestrings) != type(None) or min_target_len > 0 or max_target_len > 0:
        inter = pols.boundary.buffer(snap_dist_close/5,cap_style='square',join_style='mitre').intersection(target_geom)
        pols = pols[inter.isna() + inter.is_empty == 0].reset_index(drop=True)

        if min_target_len > 0 or max_target_len > 0:
            inter = inter[inter.isna() + inter.is_empty == 0].reset_index(drop=True)
            if min_target_len > 0:
                pols = pols[inter.length >= min_target_len].reset_index(drop=True)
            
            if max_target_len > 0:
                if min_target_len > 0:
                    inter = inter[inter.length >= min_target_len].reset_index(drop=True)

                pols = pols[inter.length <= max_target_len].reset_index(drop=True)

    if len(pols) == 0:
        return gpd.GeoSeries([],crs=orig_crs)


    pols = pols[pols.is_valid].reset_index(drop=True) 

    if isolated_pols:
        pols = create_isolated_pols(pols, prefer_small_pols=prefer_small_pols, min_intersection_length=min_target_len/4)

    if len(pols) == 0:
        return gpd.GeoSeries([],crs=orig_crs)
    else:
        pols = build_geometries(pols,crs=utm_crs)

        if type(pols) == type(None):
            print("Warning. No valid geometry found")
            return gpd.GeoSeries([],crs=orig_crs)
        else:
            pols = pols.buffer(-snap_dist_close,cap_style='square',join_style='mitre')
            pols = pols[pols.is_valid].reset_index(drop=True)
            pols = pols.buffer(snap_dist_close,cap_style='square',join_style='mitre')
            pols = pols[pols.is_valid].reset_index(drop=True)
            pols = shapely.unary_union(pols)
            pols = build_geometries(shapely.get_parts(pols),crs=utm_crs)
            if type(pols) == type(None):
                return gpd.GeoSeries([],crs=orig_crs)
            
            pols = pols[pols.is_valid].reset_index(drop=True)
            #pols = round_coordinates(pols,decimals,has_z)

            if type(bounds) == type(None):
                return pols.to_crs(orig_crs)
            else:
                pols = intersect_geoseries(pols,bounds_orig)
                if type(pols) == type(None):
                    return gpd.GeoSeries([],crs=orig_crs)
                else:
                    pols = pols.buffer(-snap_dist_close,cap_style='square',join_style='mitre')
                    pols = pols[pols.is_valid].reset_index(drop=True)
                    pols = pols.buffer(snap_dist_close,cap_style='square',join_style='mitre')
                    pols = pols[pols.is_valid].reset_index(drop=True)
                    pols = shapely.unary_union(pols)
                    pols = build_geometries(shapely.get_parts(pols),crs=utm_crs)
                    if type(pols) == type(None):
                        return gpd.GeoSeries([],crs=orig_crs)
                    #pols = round_coordinates(pols,decimals,has_z)
                    pols = pols[pols.is_valid].reset_index(drop=True)
                    return pols.to_crs(orig_crs)

