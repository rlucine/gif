''' Byte-level GIF library.
    The goal is to interface with the GIF format rather than edit the images,
    therefore each class has generally public attributes and is intended to be
    used as a struct. However, some quick functions are included due to the
    relative efficiency.
'''

## Sources cited:
# * matthewflickinger.com/lab/whatsinagif
# * onicos.com/staff/iz/formats/gif.html
# * w3.org/Graphics/GIF/spec-gif89a.txt

#--- Included modules ---
from base64 import encodebytes as base64encode
from struct import pack, unpack, calcsize
from io import BytesIO

#--- Constants ---
BLOCK_HEADER = 0x21
IMAGE_HEADER = 0x2C
BLOCK_FOOTER = 0
GRAPHIC_HEADER = 0xF9
GRAPHIC_FOOTER = 0
GRAPHIC_SIZE = 4
COMMENT_HEADER = 0xFE
TEXT_HEADER = 0x1
TEXT_SIZE = 12
APPLICATION_HEADER = 0xFF
APPLICATION_SIZE = 11
GIF_HEADER = "GIF"
GIF87a = "87a"
GIF89a = "89a"
GIF_FOOTER = 0x3B

COLORS_MAX = 256
COLOR_PAD = b"\0\0\0"

#================================================================
# Error classes
#================================================================
class GifFormatError(RuntimeError):
    '''Raised when invalid format is encountered'''
    pass

#================================================================
# Bit-level operations
#================================================================
class BitReader(object):
    '''Reads bits from a byte string'''
    
    __slots__ = [
        "_str",
        "_ptr",
        "_len",
    ]

    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self, byte_string):
        '''Initialize the reader with a complete byte string'''
        if not isinstance(byte_string, bytes):
            raise TypeError("Requires bytelike object")
        self._str = byte_string
        self._ptr = 0
        self._len = len(byte_string) * 8
    
    #------------------------------------------------
    # Bit operations
    #------------------------------------------------
    def read(self, amount):
        '''Read bits from the byte string and returns int'''
        #--- Initialize indices ---
        byte_start, start = divmod(self._ptr, 8)
        byte_end, end = divmod(min(self._ptr+amount, self._len), 8)
        #Error check
        if byte_start > self._len:
            return 0
        
        #--- Read bits ---
        if byte_start == byte_end:
            #Reading from one byte
            byte = self._str[byte_start]
            if start:
                byte >>= start
            byte &= ~(-1 << (end - start))
            #Adjust pointer
            self._ptr = (byte_end << 3) | end
            bit_str = byte
        else:
            #Reading from many bytes
            bit_str = 0
            bit_index = 0
            i = byte_start
            #Read remaining piece of the start
            if start:
                bit_str |= self._str[i] >> start
                bit_index += (8 - start)
                i += 1
            #Grab entire bytes if necessary
            while i < byte_end:
                bit_str |= (self._str[i] << bit_index)
                bit_index += 8
                i += 1
            #Read beginning piece of end byte
            if end:
                byte = self._str[i] & (~(-1 << end))
                bit_str |= (byte << bit_index)
                bit_index += end
        
        #--- Update pointer ---
        self._ptr = (byte_end << 3) | end
        return bit_str

#--- Creating bytes ---
class BitWriter(object):
    '''Writes a byte string given bit inputs'''
    
    __slots__ = [
        "_bytes",
        "_ptr",
    ]
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self):
        '''Initialize the reader with a complete byte string'''
        self._bytes = bytearray()
        self._ptr = 0
        
    #------------------------------------------------
    # Writing
    #------------------------------------------------
    def write(self, num, pad=0):
        '''Write bits to the byte string'''
        #--- Set up values ---
        max_i = i = max(num.bit_length(), pad)
        length = len(self._bytes)
        
        #--- Write bits ---
        while i > 0:
            #Check if current byte exists
            byte_start, start = divmod(self._ptr, 8)
            if length <= byte_start:
                self._bytes.append(0)
                length += 1
            #Write into current byte
            next_i = max(0, i - (8-start))
            delta = (i - next_i)
            offset = max_i - i
            value = (num & (((1 << delta) - 1) << offset)) >> offset
            self._bytes[byte_start] |= (value << start)
            #Increment pointers
            self._ptr += delta
            i = next_i
        return None
        
    #------------------------------------------------
    # Accessing data
    #------------------------------------------------
    def to_bytes(self):
        '''Returns the bytes'''
        return bytes(self._bytes)
        
#================================================================
# Stream unpacking
#================================================================
def stream_unpack(fmt, stream):
    '''Unpack the next struct from the stream'''
    ret = unpack(fmt, stream.read(calcsize(fmt)))
    if len(ret) == 1:
        return ret[0]
    return ret

#================================================================
# Block compression algorithms
#================================================================
def block_split(stream):
    '''Parses through sub-blocks and returns the entire byte string'''
    ret = bytes()
    #Parse from file stream
    block_size = stream.read(1)[0]
    while block_size:
        ret += stream.read(block_size)
        block_size = stream.read(1)[0]
    return ret
    
def block_join(raw_bytes):
    '''Dump the block as bytes'''
    blocks = bytearray()
    start_ptr = 0
    length = len(raw_bytes)
    while start_ptr != length:
        end_ptr = min(start_ptr + 254, length)
        size = end_ptr - start_ptr
        blocks.append(size)
        blocks.extend(raw_bytes[start_ptr:end_ptr])
        start_ptr = end_ptr
    #Terminator
    blocks.append(0)
    return bytes(blocks)

#================================================================
# LZW compression algorithms
#================================================================
def lzw_decompress(raw_bytes, lzw_min):
    '''Decompress the LZW data and yields output'''

    #Initialize streams
    code_in = BitReader(raw_bytes)
    idx_out = []
    #Set up bit reading
    bit_size = lzw_min + 1
    bit_inc = (1 << (bit_size)) - 1
    #Initialize special codes
    CLEAR = 1 << lzw_min
    END = CLEAR + 1
    code_table_len = END + 1
    #Begin reading codes
    code_last = -1
    while code_last != END:
        #Get the next code id
        code_id = code_in.read(bit_size)
        #Check the next code
        if code_id == CLEAR:
            #Reset size readers
            bit_size = lzw_min + 1
            bit_inc = (1 << (bit_size)) - 1
            code_last = -1
            #Clear the code table
            code_table = [-1] * code_table_len
            for x in range(code_table_len):
                code_table[x] = (x,)
        elif code_id == END:
            #End parsing
            break
        elif code_id < len(code_table) and code_table[code_id] is not None:
            current = code_table[code_id]
            #Table has code_id - output code
            idx_out.extend(current)
            k = (current[0],)
        elif code_last not in (-1, CLEAR, END):
            previous = code_table[code_last]
            #Code not in table
            k = (previous[0],)
            idx_out.extend(previous + k)
        #Check increasing the bit size
        if len(code_table) == bit_inc and bit_size < 12:
            bit_size += 1
            bit_inc = (1 << (bit_size)) - 1
        #Update the code table with previous + k
        if code_last not in (-1, CLEAR, END):
            code_table.append(code_table[code_last] + k)
        code_last = code_id
    return idx_out

def lzw_compress(indices, lzw_min):
    ''' A more optimized compression algorithm that uses a hash
        instead of a list
    '''

    #Init streams
    idx_in = map(lambda x: (x,), indices)
    bin_out = BitWriter()
    idx_buf = next(idx_in)
    #Init special codes
    CLEAR = 1 << lzw_min
    END = CLEAR + 1
    code_table_len = max(indices)+1
    #Set up bit reading
    bit_size = lzw_min + 1
    bit_inc = (1 << bit_size) - 1
    if not bit_size < 13:
        raise ValueError("Bad minumum for LZW")
    #Init code table
    index = END + 1
    code_table = dict(((x,), x) for x in range(code_table_len))
    
    #Begin with the clear code
    bin_out.write(CLEAR, bit_size)
    for k in idx_in:
        if (idx_buf + k) in code_table:
            idx_buf += k
        else:
            #Output just index buffer
            try:
                code_id = code_table[idx_buf]
            except:
                raise
            bin_out.write(code_id, bit_size)

            #Update code table
            code_table[idx_buf + k] = index
            index += 1
            idx_buf = k
            
            #Check if code table should grow
            if index-1 > bit_inc:
                if bit_size < 12:
                    bit_size += 1
                    bit_inc = (1 << bit_size) - 1
                else:
                    #Send clear code
                    bin_out.write(CLEAR, 12)
                    #Reset bit size
                    bit_size = lzw_min + 1
                    bit_inc = (1 << bit_size) - 1
                    #Reset the code table
                    code_table = dict(((x,), x) for x in range(code_table_len))
                    #Reset index
                    index = END + 1
    #Done
    bin_out.write(code_table[idx_buf], bit_size)
    #Output end-of-information code
    bin_out.write(END, bit_size)
    #BitWriter naturally pads with 0s, but now need blocking
    return bin_out.to_bytes()
    
#================================================================
# GIF component base class
#================================================================
class GifBlock(object):
    '''Base class for GIF blocks'''
    
    __slots__ = []
    
    _deprecated = False
    _version = GIF87a
    
    #------------------------------------------------
    # Check for deprecation
    #------------------------------------------------
    @classmethod
    def deprecated(cls):
        '''Check if this extension is deprecated'''
        return cls._deprecated
        
    @classmethod
    def version(cls):
        '''Get the version required for this block'''
        return cls._version
    
#================================================================
# GIF components : Image block
#================================================================
class ImageBlock(GifBlock):
    ''' Initializes a GIF image block from the file
    '''
    
    __slots__ = [
        "_x",
        "_y",
        "_width",
        "_height",
        "_interlace",
        "_lct",
        "_lzw_min",
        "_lzw",
        "_ncolors",
    ]
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self):
        ''' Create a blank image block '''
        self._x, self._y = 0, 0
        self._width, self._height = 0, 0
        self._interlace = False
        self._lct = []
        self._lzw_min = 0
        self._lzw = b""
        self._ncolors = 0
        
    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, stream, gct):
        ''' Reads bytes from the already-open file.
            Should happen after block header 0x2c is discovered
            Requires to link with the main GIF gct, but can load without file
        '''
        ret = cls()
        
        #Unpack image descriptor
        *ret.position, ret._width, ret._height, packed_byte = stream_unpack('<4HB', stream)
        
        #Unpack the packed field
        lct_exists = (packed_byte >> 7) & 1
        ret._interlace = (packed_byte >> 6) & 1
        lct_sorted = (packed_byte >> 5) & 1
        lct_size = 2 << (packed_byte & 7)
        
        #Unpack the lct if it exists
        ret._lct = []
        if lct_exists:
            i = 0
            while i < lct_size:
                ret._lct.append(stream_unpack('3B', stream))
                i += 1
        
        #Unpack actual image data
        ret._lzw_min = stream_unpack('B', stream)
        ret._lzw = block_split(stream)
        
        #Recall number of colors
        ret._ncolors = len(gct)
        return ret
        
    #------------------------------------------------
    # Dimensions
    #------------------------------------------------
    @property
    def position(self):
        '''Get the image position'''
        return self._x, self._y
        
    @position.setter
    def position(self, pos):
        '''Update the position'''
        x, y = pos
        if x >= 0 and y >= 0:
            self._x, self._y = x, y
        else:
            raise GifFormatError("Negative coordinates not allowed")
        return
        
    @property
    def width(self):
        '''Get the image block width'''
        return self._width
        
    @property
    def height(self):
        '''Get the image height'''
        return self._height
        
    #------------------------------------------------
    # Image properties
    #------------------------------------------------
    @property
    def interlace(self):
        '''Check if image is interlaced'''
        return bool(self._interlace)

    @property
    def lct(self):
        '''Get the local color table'''
        return self._lct
        
    @lct.setter
    def lct(self, new):
        '''Set the lct to a new table'''
        if self._lct and len(new) < len(self._lct):
            #New color table will truncate colors
            raise GifFormatError("LCT too small (%d colors), need %d" % (len(self._lct). len(new)))
        elif len(new) < self._ncolors:
            #New local color table cannot replace global color table
            raise GifFormatError("Cannot convert block to LCT, too small")
        #Replace the GCT
        self.lct = new
    
    #------------------------------------------------
    # Color table conversion
    #------------------------------------------------
    def update_gct(self, mapper):
        '''Make this image start using the new GCT'''

        #Generate new table
        decompressed = self.decompress()
        
        def generator():
            '''Generate the output indices'''
            for old_index in decompressed:
                yield mapper[old_index]
            return
            
        #Compute new lzw minimum
        out = list(generator())
        self._lzw_min = max(2, max(out).bit_length())
        #Compress the LZW data
        self.compress(out)
    
    def convert_to_lct(self, gct):
        '''Convert this image and the LZW data to use a local color table'''
        #Already has LCT
        if self._lct:
            return
        
        #Generate new table
        mapper = {}
        lct = []
        decompressed = self.decompress()
        
        def generator():
            '''Generate the output indices'''
            for gct_index in decompressed:
                #Capture mapping for GCT indices
                if gct_index not in mapper:
                    #Map gct index to lct index
                    mapper.setdefault(gct_index, len(lct))
                    #Map lct index to color
                    lct.append(gct[gct_index])
                yield mapper[gct_index]
            return
            
        #Set the lct
        self._lct = lct
        #Compute new lzw minimum
        out = list(generator())
        self._lzw_min = max(2, max(mapper.values()).bit_length())
        #Compress the LZW data
        self.compress(out)
        #Update number of colors
        self._ncolors = max(out)
        
    def convert_to_gct(self, gct):
        '''Convert this image so it uses the gct'''
        try:
            mapper = {lct_index: gct.index(color) for lct_index, color in enumerate(self.lct)}
            self.update_gct(mapper)
        except ValueError:
            raise GifFormatError("LCT not a subset of GCT, cannot convert") from None
        return
        
    #------------------------------------------------
    # LZW data
    #------------------------------------------------
    @property
    def lzw(self):
        '''Get the compressed LZW data'''
        return self._lzw

    def decompress(self):
        '''Decodes the LZW data'''
        return lzw_decompress(self._lzw, self._lzw_min)

    def compress(self, indices):
        '''Replace the LZW data'''
        self._lzw = lzw_compress(indices, self._lzw_min)
        
    #------------------------------------------------
    # Encoding
    #------------------------------------------------
    def encode(self):
        '''Returns the bytes of the image block
        Ostensibly, should return its own input from the file'''
        out = bytearray()
        out.append(IMAGE_HEADER)
        
        #Pack the image descriptor
        out.extend(pack('<4H', self._x, self._y, self._width, self._height))
        
        #Get packed fields
        lct_exists = bool(self._lct)
        lct_sorted = False
        
        #Construct the packed field
        packed_byte = (lct_exists << 7)
        packed_byte |= ((self._interlace & 1) << 6)
        packed_byte |= (lct_sorted << 5)
        
        #We don't need no stinking math.log
        packed_size = 0
        temp = len(self._lct) - 1
        while temp > 0:
            temp >>= 1
            packed_size += 1
        packed_byte |= ((packed_size - 1) & 7)
        #Pack the packed field
        out.append(packed_byte)

        #Pack the lct if it exists
        lct_size = 2 << (packed_size - 1) if self._lct else 0
        if lct_exists:
            i = 0
            for color in self._lct:
                out.extend(pack('3B', *color))
                i += 1
            assert i <= lct_size
            while i < lct_size:
                out.extend(COLOR_PAD)
                i += 1
        
        #Pack the lzw data
        out.append(self._lzw_min)
        out.extend(block_join(self._lzw))
        return bytes(out)
        
#================================================================
# Base class for extensions
#================================================================
class ExtensionBlock(GifBlock):
    '''Base class for all GIF extension blocks'''
    
    __slots__ = []

#================================================================
# GIF components : Graphic Control Extension block
#================================================================
class GraphicExtension(ExtensionBlock):
    ''' Initialize the graphic extension block from the file
        There can only be one per image block, but there can be arbitrarily many
        within the entire GIF
    '''
    
    __slots__ = [
        "_trans",
        "_index",
        "_delay",
        "_disposal",
        "_userin",
        "_gct",
    ]
    
    #Required version
    _version = GIF89a
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self):
        '''Load the graphic extension block'''
        self._trans = False
        self._index = 0
        self._delay = 0
        self._disposal = 0
        self._userin = False
        
    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, stream):
        ''' Reads bytes from already open file.
            Should happen after block header 0x21f9 is discovered
        '''
        ret = cls()
        
        #Unpack graphics extension block
        block_size, packed_byte = stream_unpack('2B', stream)
        if block_size != GRAPHIC_SIZE:
            raise GifFormatError("Bad graphic extension size")
        
        #Unpack the packed byte
        ret._disposal = (packed_byte >> 2) & 0x7
        ret._userin = (packed_byte >> 1) & 1
        ret._trans = packed_byte & 1
        
        #Unpack extension block
        ret._delay, ret._index, footer = stream_unpack('<HBB', stream)
        
        #Unpack block footer
        if not footer == BLOCK_FOOTER:
            raise GifFormatError("Bad graphic extension footer")
        return ret
        
    #------------------------------------------------
    # Accessors
    #------------------------------------------------
    @property
    def trans(self):
        '''Get the index of transparency, or None if nontransparent'''
        if self._trans:
            return self._index
        return None
        
    @trans.setter
    def trans(self, value):
        '''Set the transparent color'''
        if value is None:
            self._trans = False
            self._index = 0
        else:
            self._trans = True
            self._index = value
        return
    
    @property
    def delay(self):
        '''Gets the actual delay time in milliseconds'''
        return self._delay / 100
        
    @delay.setter
    def delay(self, value):
        '''Set the delay time in milliseconds'''
        self._delay = int(value * 100)
    
    #------------------------------------------------
    # Mystery properties
    #------------------------------------------------
    @property
    def disposal(self):
        '''Get the disposal method'''
        return self._disposal
        
    @property
    def user_input(self):
        '''Check if the user input flag is set'''
        return bool(self._userin)
        
    #------------------------------------------------
    # Encoding
    #------------------------------------------------
    def encode(self):
        '''Returns the bytes of the graphic extension block'''

        #Pack the packed byte
        packed_byte = ((self._disposal & 7) << 2)
        packed_byte |= ((self._userin & 1) << 1)
        packed_byte |= (self._trans & 1)
        
        out = bytearray([BLOCK_HEADER, GRAPHIC_HEADER, GRAPHIC_SIZE, packed_byte])
        
        #Pack the extension block
        out.extend(pack('<HBB', self._delay, self._index, GRAPHIC_FOOTER))
        return bytes(out)

#================================================================
# GIF components : Comment extension block
#================================================================
class CommentExtension(ExtensionBlock):
    ''' Initialize the comment extension from the file
        There can be arbitrarily many of these in one GIF
        Don't use this because it's useless
    '''
    
    #Require version
    _deprecated = True
    _version = GIF89a
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self, comment):
        '''Initialize extension with comment'''
        self._comment = comment
        
    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, stream):
        ''' Reads bytes from the already open file.
            Should happen after block header 0x21fe is found
        '''
        comment = block_split(stream).decode("ascii")
        return cls(comment)
        
    #------------------------------------------------
    # Properties
    #------------------------------------------------
    @property
    def comment(self):
        '''Get the comment text'''
        return self._comment
        
    @comment.setter
    def comment(self, value):
        '''Set the comment to a new string'''
        self._comment = str(value)

    #------------------------------------------------
    # Encoding
    #------------------------------------------------
    def encode(self):
        '''Returns the bytes of the comment extension block'''
        out = bytes([BLOCK_HEADER, COMMENT_HEADER])
        out += block_join(self._comment.encode("ascii"))
        return out
        
#================================================================
# GIF components : Plain Text Extension
#================================================================
class PlainTextExtension(ExtensionBlock):
    ''' Initialize the plain text extension from the file
        There can be arbitrarily many of these in one GIF
        This extension is probably deprecated
    '''
    
    __slots__ = [
        "_text",
        "_fg",
        "_bg",
        "_gridx",
        "_gridy",
        "_gridw",
        "_gridh",
        "_cellw",
        "_cellh",
    ]
    
    #Class is deprecated
    _deprecated = True
    _version = GIF89a
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self):
        ''' Construct extension given properties '''
        self._text = ""
        self._fg, self._bg = 0, 0
        self._gridx, self._gridy = 0, 0
        self._gridw, self._gridh = 0, 0
        self._cellw, self._cellh = 0, 0
        
    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, stream):
        ''' Reads bytes from the already open file.
            Should happen after block header 0x2101 is found
        '''
        ret = cls()
        
        #Read the block size
        block_size = stream_unpack('B', stream)
        if block_size != TEXT_SIZE:
            raise GifFormatError("Corrupt plaintext extension")
        
        #Read information
        ret._gridx, ret._gridy, *rest = stream_unpack('<4H4B', stream)
        ret._gridw, ret._gridh, *rest = rest
        ret._cellw, ret._cellh, *rest = rest
        fg, bg = rest
        ret._text = block_split(stream).decode("ascii")
        
        return ret
        
    #------------------------------------------------
    # Properties
    #------------------------------------------------
    @property
    def text(self):
        '''Get the string being displayed'''
        return self._text

    def insert_text(self, text, char_width, char_height):
        '''Set the text associated with this display'''
        self._text = text = str(text)
        
        #Set up the character cells
        self._cellw = char_width
        self._cellh = char_height
        
        #Set up the gridding of monospaced characters
        self._gridw = char_width * len(text)
        self._gridh = char_height
        
    @property
    def position(self):
        '''Get the text position'''
        return self._gridx, self._gridy
        
    @position.setter
    def position(self, value):
        '''Set the grid position'''
        x, y = value
        if x >= 0 and y >= 0:
            self._gridx, self._gridy = x, y
        else:
            raise GifFormatError("Coordinates cannot be negative")
        return
        
    @property
    def cell(self):
        '''Get the cell dimensions'''
        return self._cellw, self._cellh
        
    @property
    def grid(self):
        '''Get the grid dimensions'''
        return self._gridw, self._gridh
    
    #------------------------------------------------
    # Coloring
    #------------------------------------------------
    @property
    def foreground(self):
        '''Get the foreground color index'''
        return self._fg
        
    @foreground.setter
    def foreground(self, value):
        '''Set the foreground color'''
        self._fg = value
        
    @property
    def background(self):
        '''Get the background color index'''
        return self._bg
        
    @background.setter
    def background(self, value):
        '''Set the background color'''
        self._bg = value
    
    #------------------------------------------------
    # Encoding
    #------------------------------------------------
    def encode(self):
        '''Returns the bytes of the plain text extension block'''
        out = bytearray([BLOCK_HEADER, TEXT_HEADER, TEXT_SIZE])
        
        #Pack the grid position
        out.extend(pack("<4H", self._gridx, self._gridy, self._gridw, self._gridh))
        
        #Pack the cell properties and colors
        out.extend(pack("4B", self._cellw, self._cellh, self._fg, self._bg))
        
        #Pack the text data
        out.extend(block_join(self._text.encode("ascii")))
        return bytes(out)
        
#================================================================
# GIF components : Application Extension
#================================================================
class ApplicationExtension(ExtensionBlock):
    ''' Initialize the application extension from the file
        This is probably deprecated.
    '''
    
    __slots__ = [
        "_ident",
        "_auth",
        "_data",
    ]
    
    #Class is deprecated
    _deprecated = True
    _version = GIF89a
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self):
        ''' Initialize the block from its paramaters'''
        self._ident = ""
        self._auth = 0
        self._data = b""

    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, stream):
        ''' Decode a new application extension from the given file
            Reads bytes from the already open file.
            Should happen after block header 0x21ff is found
        '''
        ret = cls()
        
        #Read the block size
        block_size = stream_unpack('B', stream)
        if not block_size == APPLICATION_SIZE:
            raise GifFormatError("Corrupt application extension")
        
        #Read the application identifier
        ret._ident = stream.read(8).decode("ascii")
        
        #Read the application identifier
        ret._auth = int.from_bytes(stream.read(3), "little")
        
        #Read the application data (bytes)
        ret._data = block_split(stream)
        
        return ret
        
    #------------------------------------------------
    # Properties
    #------------------------------------------------
    @property
    def application(self):
        '''Get the application owning this extension'''
        return self._ident
        
    @property
    def auth_code(self):
        '''Get the authentication code'''
        return self._auth
        
    #------------------------------------------------
    # Interaction
    #------------------------------------------------
    def read(self, application, auth_code):
        '''Return the data only if the correct application and auth code are passed'''
        if self.application == application and self.auth_code == auth_code:
            #it offsers some protection, but basically can just grab the data
            #Goal is to have some sort of protocol for reading
            return self._data
        raise RuntimeError("Failed to authenticate application")

    #------------------------------------------------
    # Encoding
    #------------------------------------------------
    def encode(self):
        '''Returns the bytes of the application extension block'''
        out = bytes([BLOCK_HEADER, APPLICATION_HEADER, APPLICATION_SIZE])
        out += format(self._ident, "<8").encode("ascii")
        out += format(self._auth, "<3").encode("ascii")
        out += block_join(self._data)
        return out

#================================================================
# GIF Loading
#================================================================
class Gif(object):
    '''Opens the gif image'''
    
    __slots__ = [
        "_width",
        "_height",
        "_bgcolor",
        "_aspect",
        "_res",
        "_gct",
        "_blocks",
    ]
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self, src=None):
        '''Opens the GIF image'''
        
        #Check null init
        if src is None:
            self._width = 0
            self._height = 0
            self._bgcolor = 0
            self._aspect = 0
            self._res = 7
            self._gct = []
            self._blocks = []
            return
        
        #Check where loading from
        if isinstance(src, str):
            stream = open(filename, 'rb')
        else:
            stream = BytesIO(data)
        
        #Check header
        header = stream.read(3).decode("ascii")
        if header != GIF_HEADER:
            raise GifFormatError("Bad header: %s" % (GIF_HEADER, header))
            
        #Check version
        version = stream.read(3).decode("ascii")
        if version not in (GIF87a, GIF89a):
            raise GifFormatError("Bad version: %s" % version)

        #Check image descriptor
        self._width, self._height, *rest = stream_unpack("<2H3B", stream)
        packed_byte, self._bgcolor, self._aspect = rest
        
        #Unpack the packed byte
        gct_exists = (packed_byte >> 7) & 1
        self._res = (packed_byte >> 4) & 7 #Bits per pixel?
        gct_sorted = (packed_byte >> 3) & 1
        gct_size = 2 << (packed_byte & 7)
            
        #Unpacking the GCT
        self._gct = []
        if gct_exists:
            i = 0
            while i < gct_size:
                #color is a tuple (r, g, b)
                self._gct.append(stream_unpack('3B', stream))
                i += 1
        
        #Parse remaining blocks until reach the trailer
        self._blocks = []
        head = ord(stream.read(1))
        
        while head != GIF_FOOTER:
            
            if head == BLOCK_HEADER:
                #Discovered extension block
                label = ord(stream.read(1))
                if label == GRAPHIC_HEADER:
                    #Error check for invalid configuration
                    if self._blocks and isinstance(self._blocks[-1], GraphicExtension):
                        raise GifFormatError("Two consecutive graphic extension blocks")
                    #Load new header
                    block = GraphicExtension.decode(stream)
                elif label == COMMENT_HEADER:
                    #Load new comment
                    block = CommentExtension.decode(stream)
                elif label == TEXT_HEADER:
                    #Load new text
                    block = PlainTextExtension.decode(stream)
                elif label == APPLICATION_HEADER:
                    #Load new application data
                    block = ApplicationExtension.decode(stream)
                else:
                    raise GifFormatError("Invalid extension label: %x" % label)

            elif head == IMAGE_HEADER:
                #Discovered image data
                block = ImageBlock.decode(stream, self._gct)
            else:
                #Invalid block
                raise GifFormatError("Invalid block header: %x" % head)
            
            #Save new block in blocks list
            self._blocks.append(block)
                
            #Done with this block - get next header
            head = ord(stream.read(1))

        #Done with all blocks
        return
        
    #------------------------------------------------
    # Accessors
    #------------------------------------------------
    def version(self):
        '''Get the GIF version'''
        versions = [b.version() for b in self._blocks]
        if GIF89a in versions:
            return GIF89a
        return GIF87a
    
    @property
    def width(self):
        '''Get the GIF width'''
        return self._width
        
    @property
    def height(self):
        '''Get the GIF height'''
        return self._height
        
    @property
    def color_resolution(self):
        '''Get the color resolution'''
        return self._res + 1
        
    @property
    def aspect_ratio(self):
        '''Get the aspect ratio'''
        #According to matthewflickinger
        return ((self._aspect + 15) >> 6) if self._aspect else 0
        
    #------------------------------------------------
    # Background color
    #------------------------------------------------
    @property
    def background(self):
        '''Get the background color'''
        if self._gct:
            #GCT exists, so background color exists
            return self._bgcolor
        return None
        
    @background.setter
    def background(self, value):
        '''Sets the background color'''
        self._bgcolor = value
        
    #------------------------------------------------
    # Blocks accessor
    #------------------------------------------------
    @property
    def blocks(self):
        '''Get the GIF blocks'''
        return self._blocks

    def blocks_filter(self, block_cls):
        '''Get all blocks of the given class'''
        return [b for b in self._blocks if isinstance(b, block_cls)]
        
    #------------------------------------------------
    # GCT modification
    #------------------------------------------------
    @property
    def gct(self):
        '''Get the GCT'''
        return self._gct
        
    @gct.setter
    def gct(self, new):
        '''Replace the GCT with a new table'''
        if len(new) < len(self._gct):
            #New color table will truncate colors
            raise GifFormatError("GCT too small (%d colors), need %d" % (len(self._gct). len(new)))
        #Replace the GCT
        self._gct = new
        
    #------------------------------------------------
    # GIF encoding
    #------------------------------------------------
    def encode(self):
        '''Pack the entire GIF image'''

        #Pack name and version (standardize to 89a)
        out = bytearray((GIF_HEADER + self.version()).encode("ascii"))
        
        #Pack header attributes
        out.extend(pack("<HH", self._width, self._height))
        
        #Get packed byte fields
        gct_exists = bool(self._gct)
        
        #Construct the packed byte
        packed_byte = (gct_exists << 7)
        packed_byte |= ((self._res & 7) << 4)
        packed_byte |= (False << 3)
        
        #Determine packed GCT size
        packed_size = 0
        temp = len(self._gct) - 1 #Off by one error
        while temp > 0:
            temp >>= 1
            packed_size += 1
        packed_byte |= ((packed_size - 1) & 7)
        
        #Pack the packed byte and other fields
        out.extend(pack("3B", packed_byte, self._bgcolor, self._aspect))
        
        #Pack the gct
        gct_size = 2 << (packed_size - 1)
        if gct_exists:
            #Pack entire GCT
            i = 0
            for color in self._gct:
                out.extend(pack('3B', *color))
                i += 1
            
            assert i <= gct_size
            #Pad with null colors
            while i < gct_size:
                out.extend(COLOR_PAD)
                i += 1
                
        #Pack the ramaining blocks (these supply their own headers)
        for block in self._blocks:
            out.extend(block.encode())
            
        #Pack the trailer
        out.extend(GIF_FOOTER.to_bytes(1, "little"))
        return bytes(out)

    #------------------------------------------------
    # Mutation
    #------------------------------------------------
    def convert_gif87a(self):
        '''Convert the image to GIF87a format by discarding other blocks'''
        i = 0
        while i < len(self._blocks):
            if self._blocks[i].version() != GIF87a:
                del self._blocks[i]
            else:
                i += 1
        return
    
    def optimize(self):
        ''' Optimize the size of this image by removing extension blocks
            and resizing color tables.
        '''
        initial_size = len(self.encode())
        
        #Remove useless blocks
        i = 0
        while i < len(self._blocks):
            block = self._blocks[i]
            if self._blocks[i].deprecated():
                del self._blocks[i]
            else:
                i += 1
                
        #Optimize the GCT
        used = set()
        for block in self.blocks:
            #Images that use GCT
            if isinstance(block, ImageBlock) and not block.lct:
                used.update(block.decompress())
            elif isinstance(block, GraphicExtension):
                if block.trans:
                    used.add(block.trans)
            elif isinstance(block, PlainTextExtension):
                used.add(block.foreground)
                used.add(block.background)
        
        #Get mapping from old GCT to new
        used_list = list(enumerate(used))
        mapper = {old_index: new_index for new_index, old_index in used_list}
        for block in self._blocks:
            if isinstance(block, ImageBlock):
                block.update_gct(mapper)

        #Store the new gct
        self._gct = [self._gct[i] for _, i in used_list]

        #Return delta in size
        final_size = len(self.encode())
        return initial_size - final_size

#================================================================
# Testing
#================================================================
if __name__ == "__main__":

    from time import time

    def test(path):
        '''Unit test on one gif image'''
        print("Testing %s" % path)
        #Time the opening of the GIF image
        ti = time()
        g = Gif(filename=path)
        tf = time()
        print("Opened in %f seconds" % (tf - ti))
        ti = tf
        #Time the decompression of the image block
        block = g.blocks_filter(ImageBlock)[0]
        d = list(block.decompress())
        tf = time()
        print("Decompressed in %f seconds" % (tf - ti))
        #Ensure the image block is actually valid
        assert len(d) == (g.height * g.width)
        ti = tf
        #Time the recompression of the image block
        block.compress(d)
        #Calculate differences in the image block
        tf = time()
        print("Compressed in %f seconds" % (tf - ti))
        #Uncompress the compress
        de = list(block.decompress())
        assert d == de
        #Check encoding
        ti - tf
        g.encode()
        tf = time()
        print("Encoded in %f seconds!" % (tf - ti))
        #Done!
        print("Passed test!\n")

    """test(r"..\dev\image-test\sample_1.gif")
    test(r"..\dev\image-test\writegif.gif")
    test(r"..\dev\image-test\bitdepth1.gif")
    test(r"..\dev\image-test\bitdepth2.gif")
    test(r"..\dev\image-test\bitdepth4.gif")
    test(r"..\dev\image-test\animated.gif")
    test(r"..\dev\image-test\test.GIF")
    test(r"..\dev\image-test\audrey.gif")
    test(r"..\dev\image-test\audrey_big.gif")
    test(r"..\dev\image-test\audrey_hq.gif")"""

    g = Gif(filename="../dev/image-test/audrey_big.gif")
    gfx, image = g.blocks
    print(g.optimize())
    g.convert_gif87a()
    with open("../dev/image-test/opt_test.gif", "wb") as file:
        file.write(g.encode())
    
