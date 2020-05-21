from osgeo import ogr, gdal
import matplotlib.pyplot as plt


InputVector = './data/testarea2_new/shapefile/tree_species.shp'

OutputImage = './data/testarea2_new/sg_quac_test2_gt.tif'

RefImage = './data/testarea2_new/sg_quac_test2.tif'

gdalformat = 'GTiff'
datatype = gdal.GDT_Byte
# burnVal = 9

Image = gdal.Open(RefImage, gdal.GA_ReadOnly)
print(type(Image))
print(Image.RasterXSize)
print(Image.RasterYSize)

Shapefile = ogr.Open(InputVector)
Shapefile_layer = Shapefile.GetLayer()

print("Rasterising shapefile ...")

Output = gdal.GetDriverByName(gdalformat).Create(
    OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype,
    options=['COMPRESS=DEFLATE'])

Output.SetProjection(Image.GetProjectionRef())
Output.SetGeoTransform(Image.GetGeoTransform())

Band = Output.GetRasterBand(1)
Band.SetNoDataValue(0)
gdal.RasterizeLayer(
    Output, [1], Shapefile_layer,
    burn_values=[100],
    options=["ATTRIBUTE=sp_num"])

Band = None
Output = None
Image = None
Shapefile = None

OutputImage_ = gdal.Open(OutputImage).ReadAsArray()
print(OutputImage_)
import numpy as np

a = np.unique(OutputImage_)
print(a)

plt.imshow(OutputImage_, cmap='gray')
plt.colorbar()

plt.show()