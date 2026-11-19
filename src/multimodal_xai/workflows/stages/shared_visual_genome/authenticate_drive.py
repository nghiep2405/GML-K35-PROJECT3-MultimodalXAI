assert vg_deps_ready
from gdrive_fsspec import GoogleDriveFileSystem

import os
VG_DRIVE_FOLDER_ID = os.environ["VG_DRIVE_FOLDER_ID"]
vg_drive = GoogleDriveFileSystem(
    token="browser",
    auth_kwargs={"use_local_webserver": False},
    root_file_id=VG_DRIVE_FOLDER_ID,
    use_listings_cache=False,
    skip_instance_cache=True,
)
print("Drive root entries:")
for vg_item in vg_drive.ls("", detail=True):
    print(vg_item["name"], vg_item["type"])
vg_drive_ready = True
