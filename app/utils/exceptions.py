class ImageNotFoundError(Exception):
    """Raised when a requested image ID doesn't exist."""
    def __init__(self, image_id: str):
        self.image_id = image_id
        super().__init__(f"Image with id '{image_id}' not found")
