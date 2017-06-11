Help on module gif:

NAME
    gif

DESCRIPTION
    Byte-level GIF library.
    The goal is to interface with the GIF format rather than edit the images,
    therefore each class has generally public attributes and is intended to be
    used as a struct. However, some quick functions are included due to the
    relative efficiency.
        
        Two command-line invocations are provided:
        * gif.py test <file.gif>
        This decompresses and recompresses the image to test the module.
        
        * gif.py optimize <input.gif> <output.gif>
        This compresses the image by removing extra colors and blocks.

CLASSES
    builtins.RuntimeError(builtins.Exception)
        GifFormatError
    builtins.object
        Gif
        GifBlock
            ExtensionBlock
                ApplicationExtension
                CommentExtension
                GraphicExtension
                PlainTextExtension
            ImageBlock
    
    class ApplicationExtension(ExtensionBlock)
     |  Initialize the application extension from the file
     |  This is probably deprecated.
     |  
     |  Method resolution order:
     |      ApplicationExtension
     |      ExtensionBlock
     |      GifBlock
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self)
     |      Initialize the block from its paramaters
     |  
     |  encode(self)
     |      Returns the bytes of the application extension block
     |  
     |  read(self, application, auth_code)
     |      Return the data only if the correct application and auth code are passed
     |  
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |  
     |  decode(stream) from builtins.type
     |      Decode a new application extension from the given file
     |      Reads bytes from the already open file.
     |      Should happen after block header 0x21ff is found
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  application
     |      Get the application owning this extension
     |  
     |  auth_code
     |      Get the authentication code
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from GifBlock:
     |  
     |  deprecated() from builtins.type
     |      Check if this extension is deprecated
     |  
     |  version() from builtins.type
     |      Get the version required for this block
    
    class CommentExtension(ExtensionBlock)
     |  Initialize the comment extension from the file
     |  There can be arbitrarily many of these in one GIF
     |  Don't use this because it's useless
     |  
     |  Method resolution order:
     |      CommentExtension
     |      ExtensionBlock
     |      GifBlock
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, comment)
     |      Initialize extension with comment
     |  
     |  encode(self)
     |      Returns the bytes of the comment extension block
     |  
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |  
     |  decode(stream) from builtins.type
     |      Reads bytes from the already open file.
     |      Should happen after block header 0x21fe is found
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  comment
     |      Get the comment text
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from GifBlock:
     |  
     |  deprecated() from builtins.type
     |      Check if this extension is deprecated
     |  
     |  version() from builtins.type
     |      Get the version required for this block
    
    class ExtensionBlock(GifBlock)
     |  Base class for all GIF extension blocks
     |  
     |  Method resolution order:
     |      ExtensionBlock
     |      GifBlock
     |      builtins.object
     |  
     |  Class methods inherited from GifBlock:
     |  
     |  deprecated() from builtins.type
     |      Check if this extension is deprecated
     |  
     |  version() from builtins.type
     |      Get the version required for this block
    
    class Gif(builtins.object)
     |  Opens the gif image
     |  
     |  Methods defined here:
     |  
     |  __init__(self, src=None)
     |      Opens the GIF image
     |  
     |  blocks_filter(self, block_cls)
     |      Get all blocks of the given class
     |  
     |  convert_gif87a(self)
     |      Convert the image to GIF87a format by discarding other blocks
     |  
     |  encode(self)
     |      Pack the entire GIF image
     |  
     |  optimize(self)
     |      Optimize the size of this image by removing extension blocks
     |      and resizing color tables.
     |  
     |  version(self)
     |      Get the GIF version
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  aspect_ratio
     |      Get the aspect ratio
     |  
     |  background
     |      Get the background color
     |  
     |  blocks
     |      Get the GIF blocks
     |  
     |  color_resolution
     |      Get the color resolution
     |  
     |  gct
     |      Get the GCT
     |  
     |  height
     |      Get the GIF height
     |  
     |  width
     |      Get the GIF width
    
    class GifBlock(builtins.object)
     |  Base class for GIF blocks
     |  
     |  Class methods defined here:
     |  
     |  deprecated() from builtins.type
     |      Check if this extension is deprecated
     |  
     |  version() from builtins.type
     |      Get the version required for this block
    
    class GifFormatError(builtins.RuntimeError)
     |  Raised when invalid format is encountered
     |  
     |  Method resolution order:
     |      GifFormatError
     |      builtins.RuntimeError
     |      builtins.Exception
     |      builtins.BaseException
     |      builtins.object
     |  
     |  Data descriptors defined here:
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from builtins.RuntimeError:
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __new__(*args, **kwargs) from builtins.type
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from builtins.BaseException:
     |  
     |  __delattr__(self, name, /)
     |      Implement delattr(self, name).
     |  
     |  __getattribute__(self, name, /)
     |      Return getattr(self, name).
     |  
     |  __reduce__(...)
     |      helper for pickle
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __setattr__(self, name, value, /)
     |      Implement setattr(self, name, value).
     |  
     |  __setstate__(...)
     |  
     |  __str__(self, /)
     |      Return str(self).
     |  
     |  with_traceback(...)
     |      Exception.with_traceback(tb) --
     |      set self.__traceback__ to tb and return self.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from builtins.BaseException:
     |  
     |  __cause__
     |      exception cause
     |  
     |  __context__
     |      exception context
     |  
     |  __dict__
     |  
     |  __suppress_context__
     |  
     |  __traceback__
     |  
     |  args
    
    class GraphicExtension(ExtensionBlock)
     |  Initialize the graphic extension block from the file
     |  There can only be one per image block, but there can be arbitrarily many
     |  within the entire GIF
     |  
     |  Method resolution order:
     |      GraphicExtension
     |      ExtensionBlock
     |      GifBlock
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self)
     |      Load the graphic extension block
     |  
     |  encode(self)
     |      Returns the bytes of the graphic extension block
     |  
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |  
     |  decode(stream) from builtins.type
     |      Reads bytes from already open file.
     |      Should happen after block header 0x21f9 is discovered
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  delay
     |      Gets the actual delay time in milliseconds
     |  
     |  disposal
     |      Get the disposal method
     |  
     |  trans
     |      Get the index of transparency, or None if nontransparent
     |  
     |  user_input
     |      Check if the user input flag is set
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from GifBlock:
     |  
     |  deprecated() from builtins.type
     |      Check if this extension is deprecated
     |  
     |  version() from builtins.type
     |      Get the version required for this block
    
    class ImageBlock(GifBlock)
     |  Initializes a GIF image block from the file
     |  
     |  Method resolution order:
     |      ImageBlock
     |      GifBlock
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self)
     |      Create a blank image block
     |  
     |  compress(self, indices)
     |      Replace the LZW data
     |  
     |  convert_to_gct(self, gct)
     |      Convert this image so it uses the gct
     |  
     |  convert_to_lct(self, gct)
     |      Convert this image and the LZW data to use a local color table
     |  
     |  decompress(self)
     |      Decodes the LZW data
     |  
     |  encode(self)
     |      Returns the bytes of the image block
     |      Ostensibly, should return its own input from the file
     |  
     |  update_gct(self, mapper)
     |      Make this image start using the new GCT
     |  
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |  
     |  decode(stream, gct) from builtins.type
     |      Reads bytes from the already-open file.
     |      Should happen after block header 0x2c is discovered
     |      Requires to link with the main GIF gct, but can load without file
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  height
     |      Get the image height
     |  
     |  interlace
     |      Check if image is interlaced
     |  
     |  lct
     |      Get the local color table
     |  
     |  lzw
     |      Get the compressed LZW data
     |  
     |  position
     |      Get the image position
     |  
     |  width
     |      Get the image block width
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from GifBlock:
     |  
     |  deprecated() from builtins.type
     |      Check if this extension is deprecated
     |  
     |  version() from builtins.type
     |      Get the version required for this block
    
    class PlainTextExtension(ExtensionBlock)
     |  Initialize the plain text extension from the file
     |  There can be arbitrarily many of these in one GIF
     |  This extension is probably deprecated
     |  
     |  Method resolution order:
     |      PlainTextExtension
     |      ExtensionBlock
     |      GifBlock
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self)
     |      Construct extension given properties
     |  
     |  encode(self)
     |      Returns the bytes of the plain text extension block
     |  
     |  insert_text(self, text, char_width, char_height)
     |      Set the text associated with this display
     |  
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |  
     |  decode(stream) from builtins.type
     |      Reads bytes from the already open file.
     |      Should happen after block header 0x2101 is found
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  background
     |      Get the background color index
     |  
     |  cell
     |      Get the cell dimensions
     |  
     |  foreground
     |      Get the foreground color index
     |  
     |  grid
     |      Get the grid dimensions
     |  
     |  position
     |      Get the text position
     |  
     |  text
     |      Get the string being displayed
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from GifBlock:
     |  
     |  deprecated() from builtins.type
     |      Check if this extension is deprecated
     |  
     |  version() from builtins.type
     |      Get the version required for this block

DATA
    GIF87a = '87a'
    GIF89a = '89a'
    __all__ = ['GIF87a', 'GIF89a', 'GifFormatError', 'GifBlock', 'Extensio...

FILE
    c:\users\alec\desktop\gif\gif.py


