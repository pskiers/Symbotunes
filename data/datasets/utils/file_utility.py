import os


class FileUtility(object):
    @staticmethod
    def get_filename_with_postfix(directory, base_name):
        counter = 1
        while True:
            bare_name, extension = os.path.splitext(base_name)
            filename = f"{bare_name}_{counter}{extension}"
            if not os.path.exists(os.path.join(directory, filename)):
                return filename
            counter += 1
