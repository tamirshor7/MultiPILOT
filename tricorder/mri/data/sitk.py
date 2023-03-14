import SimpleITK as sitk

def FileRead(file_path, return_sitkImg = False, ImageIO=None):
    reader = sitk.ImageFileReader()
    if bool(ImageIO): reader.SetImageIO(ImageIO)
    reader.SetFileName(file_path)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image)
    if not return_sitkImg:
        return arr
    else:
        return arr, image

def FileSave(data, file_path):
    if type(data) is not sitk.Image: data = sitk.GetImageFromArray(data)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_path)
    writer.Execute(data)

def CreateSITKmage(data):
    return sitk.GetImageFromArray(data)

def Show(data, title=None, application=None, command=None, ext=None):
    if type(data) is not sitk.Image: data = sitk.GetImageFromArray(data)
    image_viewer = sitk.ImageViewer()
    if bool(title): image_viewer.SetTitle(title)
    if bool(application): image_viewer.SetApplication(application)
    if bool(command): image_viewer.SetCommand(command)
    if bool(ext): image_viewer.SetFileExtension(ext)
    image_viewer.Execute(data)