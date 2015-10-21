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
import struct
from base64 import encodebytes as base64encode

#--- Constants ---
BLOCK_HEADER = 0x21
IMAGE_HEADER = 0x2C
BLOCK_FOOTER = 0
GRAPHIC_HEADER = 0xF9
GRAPHIC_SIZE = 4
COMMENT_HEADER = 0xFE
TEXT_HEADER = 0x1
TEXT_SIZE = 12
APPLICATION_HEADER = 0xFF
APPLICATION_SIZE = 11
GIF_HEADER = "GIF"
GIF_87A = "87a"
GIF_89A = "89a"
GIF_FOOTER = 0x3B

#================================================================
# Error classes
#================================================================
class GifFormatError(RuntimeError):
    '''Raised when invalid format is encountered'''
    pass

#================================================================
# Byte streaming class
#================================================================
class ByteStream(object):
    '''Object that takes a byte string and can read bytes as from
    a file'''
    
    __slots__ = [
        "_source",
    ]
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self, source):
        '''Initialize stream based on some source in memory'''
        if not isinstance(source, bytes):
            raise TypeError("Requires byte-like object")
        #Do this to avoid having source be a bytes_iterator
        self._source = iter(source)
    
    #------------------------------------------------
    # Reading
    #------------------------------------------------
    def read(self, amount):
        '''Read amount of bytes from the stream'''
        out = bytearray()
        try:
            while amount:
                #The generator produces int, not bytes
                out.append(next(self._source))
                amount -= 1
        except StopIteration:
            pass
        return bytes(out)

    def unpack(self, fmt):
        '''Read a new struct-formatted tuple from stream
        If only one item in tuple, return just the item'''
        size = struct.calcsize(fmt)
        temp = struct.unpack(fmt, self.read(size))
        if len(temp) == 1:
            return temp[0]
        return temp
        
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
def lzw_decompress(raw_bytes, lzw_min, color_table):
    '''Decompress the LZW data and yields output'''
    #Initialize streams
    code_in = BitReader(raw_bytes)
    idx_out = list()
    #Set up bit reading
    bit_size = lzw_min + 1
    bit_inc = (1 << (bit_size)) - 1
    #Initialize special codes
    code_table_len = len(color_table)
    CLEAR = 1 << lzw_min
    END = CLEAR + 1
    #Begin reading codes
    code_last = None
    while code_last != END:
        #Get the next code id
        code_id = code_in.read(bit_size)
        #Check the next code
        if code_id == CLEAR:
            #Reset size readers
            bit_size = lzw_min + 1
            bit_inc = (1 << (bit_size)) - 1
            code_last = None
            #Clear the code table
            code_table = [None for x in range(END + 1)]
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
        elif code_last is not None:
            previous = code_table[code_last]
            #Code not in table
            k = (previous[0],)
            idx_out.extend(previous + k)
        #Check increasing the bit size
        if len(code_table) == bit_inc and bit_size < 12:
            bit_size += 1
            bit_inc = (1 << (bit_size)) - 1
        #Update the code table with previous + k
        if code_last is not None and code_table[code_last]:
            code_table.append(code_table[code_last] + k)
        code_last = code_id
    return idx_out

def lzw_compress(indices, lzw_min, color_table):
    '''A more optimized compression algorithm that uses a hash
    instead of a list'''
    #Init streams
    idx_in = iter((x,) for x in indices)
    bin_out = BitWriter()
    idx_buf = next(idx_in)
    #Init special codes
    CLEAR = 1 << lzw_min
    END = CLEAR + 1
    #Set up bit reading
    bit_size = lzw_min + 1
    bit_inc = (1 << bit_size) - 1
    if not bit_size < 13:
        raise ValueError("Bad minumum for LZW")
    #Init code table
    code_table_len = len(color_table)
    index = END + 1
    code_table = dict(((x,), x) for x in range(code_table_len))
    
    #Begin with the clear code
    bin_out.write(CLEAR, bit_size)
    for k in idx_in:
        if (idx_buf + k) in code_table:
            idx_buf += k
        else:
            #Output just index buffer
            code_id = code_table[idx_buf]
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
# Base class for GIF blocks
#================================================================
class GifBlock(object):
    '''Base class for all GIF blocks'''
    
    __slots__ = []
    
#================================================================
# GIF components : Image block
#================================================================
class ImageBlock(GifBlock):
    '''Initializes a GIF image block from the file
    Don't even bother with private attributes, there are too many'''
    
    __slots__ = [
        "_x",
        "_y",
        "_width",
        "_height",
        "_interlace",
        "_lct",
        "_lzw_min",
        "_lzw",
        "_gct",
    ]
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self, gct, lzw='', lzw_min=0, *, lct=None, inter=0, pos=None, size=None):
        ''' Loads an image block given its properties '''
        #Create a blank image block
        self._x, self._y = pos if pos else (0, 0)
        self._width, self._height = size if size else (0, 0)
        self._interlace = bool(inter)
        self._lct = lct if lct else []
        self._lzw_min = lzw_min
        self._lzw = lzw
        #Link to external global color table
        self._gct = gct
        
    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, file, gct):
        ''' Reads bytes from the already-open file.
            Should happen after block header 0x2c is discovered
            Requires to link with the main GIF gct, but can load without file
        '''
        #Unpack image descriptor
        x, y, width, height, packed_byte = file.unpack('<4HB')
        
        #Unpack the packed field
        lct_exists = (packed_byte >> 7) & 1
        interlace = (packed_byte >> 6) & 1
        lct_sorted = (packed_byte >> 5) & 1
        lct_size = 2 << (packed_byte & 7)
        
        #Unpack the lct if it exists
        lct = []
        if lct_exists:
            i = 0
            while i < lct_size:
                color = file.unpack('3B')
                lct.append(color)
                i += 1
        
        #Unpack actual image data
        lzw_min = file.unpack('B')
        lzw = block_split(file)
        
        return cls(
            gct,
            lzw,
            lzw_min,
            lct=lct,
            inter=interlace,
            pos=(x, y),
            size=(width, height),
        )
        
    #------------------------------------------------
    # Encoding
    #------------------------------------------------
    def encode(self):
        '''Returns the bytes of the image block
        Ostensibly, should return its own input from the file'''
        out = bytes([IMAGE_HEADER])
        
        #Pack the image descriptor
        out += struct.pack('<4H', 
            self._x,
            self._y,
            self._width,
            self._height
        )
        
        #Get packed fields
        lct_exists = bool(self._lct)
        lct_sorted = self._lct == sorted(self._lct)
        
        #Construct the packed field
        packed_byte = 0
        packed_byte |= (lct_exists << 7)
        packed_byte |= ((self._interlace & 1) << 6)
        packed_byte |= (lct_sorted << 5)
        #We don't need no stinking math.log
        packed_size = 0
        temp = len(self._lct) - 1
        while temp > 0:
            temp >>= 1
            packed_size += 1
        packed_byte |= ((packed_size - 1) & 7)
        lct_size = 2 << (packed_size - 1) if self._lct else 0
        
        #Pack the packed field
        out += packed_byte.to_bytes(1, "little")
        
        #Pack the lct if it exists
        if lct_exists:
            i = 0
            for color in self._lct:
                out += struct.pack('3B', *color)
                i += 1
            assert i <= lct_size
            while i < lct_size:
                out += b"\0\0\0"
                i += 1
        
        #Pack the lzw data
        out += self._lzw_min.to_bytes(1, "little")
        out += block_join(self._lzw)
        return out

    #------------------------------------------------
    # Compression
    #------------------------------------------------
    def decompress(self):
        '''Decodes the LZW data'''
        table = self._lct if self._lct else self._gct
        return lzw_decompress(self._lzw, self._lzw_min, table)

    def compress(self, indices):
        '''Replace the LZW data'''
        table = self._lct if self._lct else self._gct
        self._lzw = lzw_compress(indices, self._lzw_min, table)
        
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
    ]
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self, *, trans=0, index=0, delay=0, disposal=0, userin=0):
        '''Load the graphic extension block'''
        self._trans = bool(trans)
        if not 0 <= index < 256:
            raise ValueError("Index must be in [0, 256)")
        self._index = index
        self._delay = delay
        self._disposal = bool(disposal)
        self._userin = bool(userin)
        
    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, file):
        ''' Reads bytes from already open file.
            Should happen after block header 0x21f9 is discovered
        '''
        #Unpack graphics extension block
        block_size, packed_byte = file.unpack('2B')
        if block_size != GRAPHIC_SIZE:
            raise GifFormatError("Corrupt graphic extension")
        
        #Unpack the packed byte
        disposal = (packed_byte >> 2) & 0x7
        userin = (packed_byte >> 1) & 1
        trans_flag = packed_byte & 1
        
        #Unpack extension block
        delay = file.unpack('<H')
        trans_index = file.unpack('B')
        
        #Unpack block finish
        if not file.unpack('B') == BLOCK_FOOTER:
            raise GifFormatError("Corrupt graphic extension")
        return cls(
            trans=trans_flag,
            index=trans_index,
            delay=delay,
            disposal=disposal,
            userin=userin,
        )
        
    #------------------------------------------------
    # Encoding
    #------------------------------------------------
    def encode(self):
        '''Returns the bytes of the graphic extension block'''
        out = bytes([BLOCK_HEADER, GRAPHIC_HEADER,GRAPHIC_SIZE])

        #Pack the packed byte
        packed_byte = 0
        packed_byte |= ((self._disposal & 7) << 2)
        packed_byte |= ((self._userin & 1) << 1)
        packed_byte |= (self._trans & 1)
        out += packed_byte.to_bytes(1, "little")
        
        #Pack the extension block
        out += struct.pack('<HBB', self._delay, self._index, 0)
        return out

    #------------------------------------------------
    # Accessors
    #------------------------------------------------
    def trans(self):
        '''Get the index of transparency, or None if nontransparent'''
        if self._trans:
            return self._index
        return None
    
    def delay(self):
        '''Gets the actual delay time in milliseconds'''
        return self._delay / 100
        
    #------------------------------------------------
    # Usefullness test
    #------------------------------------------------
    def useful(self):
        '''Check if this extension block is doing anything'''
        return bool(self._trans | self._delay | self._disposal | self._userin)

#================================================================
# GIF components : Comment extension block
#================================================================
class CommentExtension(ExtensionBlock):
    ''' Initialize the comment extension from the file
        There can be arbitrarily many of these in one GIF
        Don't use this because it's useless
    '''
    
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
    def decode(cls, file):
        ''' Reads bytes from the already open file.
            Should happen after block header 0x21fe is found
        '''
        comment = block_split(file).decode("ascii")
        return cls(comment)
        
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
        "_data",
        "_fg",
        "_bg",
        "_gridx",
        "_gridy",
        "_gridw",
        "_gridh",
        "_cellw",
        "_cellh",
    ]
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self, data='', *, color=None, grid=None, size=None, cell=None):
        ''' Construct extension given properties '''
        self._data = data
        #Unpack colors
        self._fg, self._bg = color if color else (1, 0)
        if not(0 <= self._fg < 256 and 0 <= self._bg < 256):
            raise ValueError("Color indices must be in [0, 256)")
        #Unpack coordinates
        self._gridx, self._gridy = grid if grid else (0, 0)
        if not(self._gridx >= 0 and self._gridy >= 0):
            raise ValueError("Grid coordinates must be >= 0")
        #Unpack size
        self._gridw, self._gridh = size if size else (0, 0)
        if not(self._gridw >= 0 and self._gridh >= 0):
            raise ValueError("Grid size must be >= 0")
        #Unpack cell size
        self._cellw, self._cellh = cell if cell else (0, 0)
        if not(self._cellw >= 0 and self._cellh >= 0):
            raise ValueError("Cell size must be >= 0")
        return
        
    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, file):
        ''' Reads bytes from the already open file.
            Should happen after block header 0x2101 is found
        '''
        #Read the block size
        block_size = file.unpack('B')
        if block_size != TEXT_SIZE:
            raise GifFormatError("Corrupt plaintext extension")
        
        #Read information
        left, top = file.unpack('<2H')
        width, height = file.unpack('<2H')
        cell_width, cell_height = file.unpack('2B')
        fg, bg = file.unpack('2B')
        data = block_split(file).decode("ascii")
        
        return cls(
            data,
            color=(fg, bg),
            grid=(left, top),
            size=(width, height),
            cell=(cell_width, cell_height),
        )
    
    #------------------------------------------------
    # Encoding
    #------------------------------------------------
    def encode(self):
        '''Returns the bytes of the plain text extension block'''
        out = bytes([BLOCK_HEADER, TEXT_HEADER, TEXT_SIZE])
        
        #Pack the grid position
        out += struct.pack(
            "<4H", 
            self._gridx,
            self._gridy,
            self._gridw,
            self._gridh,
        )
        
        #Pack the cell properties and colors
        out += struct.pack("4B",
            self._cellw,
            self._cellh,
            self._fg,
            self._bg,
        )
        
        #Pack the text data
        out += block_join(self._data.encode("ascii"))
        return out
        
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
    
    #------------------------------------------------
    # Construction
    #------------------------------------------------
    def __init__(self, ident='', auth='', data=''):
        ''' Initialize the block from its paramaters'''
        self._ident = ident
        self._auth = auth
        self._data = data

    #------------------------------------------------
    # Decoding
    #------------------------------------------------
    @classmethod
    def decode(cls, file):
        ''' Decode a new application extension from the given file
            Reads bytes from the already open file.
            Should happen after block header 0x21ff is found
        '''
        #Read the block size
        block_size = file.unpack('B')
        if not block_size == APPLICATION_SIZE:
            raise GifFormatError("Corrupt application extension")
        
        #Read the application identifier
        ident = file.read(8).decode("ascii")
        
        ## This bit is sketchy:
        # * onicos says the application data block begins immediately
        #
        # * mattherflickinger says the next field is a 3-byte authentication
        #   code, and THEN the block data follows. Who knows? 
        #   Nobody uses this...
        #   Because the block size is 0xb == 0x8 + 0x3, assume the auth code
        #   is supposed to be there.
        
        #Read the application identifier
        auth = file.read(3).decode("ascii")
        
        #Read the application data
        data = block_split(file)
        
        return cls(ident, auth, data)
        
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
        "_version",
        "_width",
        "_height",
        "_bgcolor",
        "_aspect",
        "_res",
        "_gct",
        "_blocks",
    ]
    
    #------------------------------------------------
    # Gif construction
    #------------------------------------------------
    def __init__(self, data):
        '''Opens the GIF image'''
        #Load GIF from file or bytes object
        if isinstance(data, str):
            #Try opening file
            with open(data, 'rb') as temp:
                data = temp.read()
        if type(data) not in (bytes, bytearray):
            raise TypeError("Data must be bytes or filename")
        file = ByteStream(data)
        
        #--- Check header ---
        header = file.read(3).decode("ascii")
        if header != GIF_HEADER:
            raise GifFormatError(
                "Bad header, expected %s but found %s" % (
                    GIF_HEADER,
                    header
                )
            )
            
        #--- Check version ---
        version = file.read(3).decode("ascii")
        if version not in (GIF_87A, GIF_89A):
            raise GifFormatError(
                "Bad version: %s" % version
            )
        self._version = version

        #--- Unpack image descriptor ---
        self._width, self._height = file.unpack('<2H')
        packed_byte, self._bgcolor, aspect_ratio = file.unpack('3B')
        
        #Unpack the packed byte
        gct_exists = (packed_byte >> 7) & 1
        self._res = (packed_byte >> 4) & 7 #Bits per pixel?
        sorted_flag = (packed_byte >> 3) & 1
        gct_size = 2 << (packed_byte & 7)

        #Unpack pixel aspect ratio (according to matthewflickinger)
        self._aspect = (aspect_ratio + 15) >> 6 if aspect_ratio else 0
            
        #Unpacking the GCT
        self._gct = []
        if gct_exists:
            i = 0
            while i < gct_size:
                #color is a tuple (r, g, b)
                color = file.unpack('3B')
                self._gct.append(color)
                i += 1
        
        #Parse remaining blocks until reach the trailer
        self._blocks = list()
        temp = file.unpack('B')
        while temp != GIF_FOOTER:
            if temp == BLOCK_HEADER:
                #Discovered extension block
                ext_label = file.unpack('B')
                if ext_label == GRAPHIC_HEADER:
                    #Error check for invalid configuration
                    if self._blocks:
                        if isinstance(self._blocks[-1], GraphicExtension):
                            raise GifFormatError(
                                "Two consecutive graphic extension blocks"
                            )
                    #Load new header
                    self._blocks.append(GraphicExtension.decode(file))
                elif ext_label == COMMENT_HEADER:
                    #Load new comment
                    self._blocks.append(CommentExtension.decode(file))
                elif ext_label == TEXT_HEADER:
                    #Load new text
                    self._blocks.append(PlainTextExtension.decode(file))
                elif ext_label == APPLICATION_HEADER:
                    #Load new application data
                    self._blocks.append(ApplicationExtension.decode(file))
                else:
                    raise GifFormatError(
                        "Invalid extension label: %x" % ext_label
                    )
            elif temp == IMAGE_HEADER:
                #Discovered image data
                self._blocks.append(ImageBlock.decode(file, self._gct))
            else:
                #Invalid block
                raise GifFormatError("Invalid block header: %x" % temp)
            #Done with this block - get next header
            temp = file.unpack('B')
        #Done with all blocks
        return
        
    #------------------------------------------------
    # GIF encoding
    #------------------------------------------------
    def encode(self):
        '''Pack the entire GIF image'''
        out = bytes()
        
        #Pack name and version
        if self._version not in (GIF_87A, GIF_89A):
            raise GifFormatError("Bad version: %s" % self._version)
        out += (GIF_HEADER + self._version).encode("ascii")
        
        #Pack header attributes
        if self._width >= 0 and self._height >= 0:
            out += struct.pack("<HH", self._width, self._height)
        else:
            raise GifFormatError("Image dimensions must be positive")
        
        #Get packed byte fields
        gct_exists = bool(self._gct)
        gct_sorted = self._gct == sorted(self._gct)
        
        #Check color resolution
        if self._res != self._res & 7:
            raise GifFormatError(
                "Bad color res: %x, must be in [0, 7]" % self._color_res
            )
        
        #Construct the packed byte
        packed_byte = 0
        packed_byte |= (gct_exists << 7)
        packed_byte |= ((self._res & 7) << 4)
        packed_byte |= (gct_sorted << 3)
        
        #Error check GCT
        packed_size = 0
        temp = len(self._gct) - 1
        if temp > 256:
            raise GifFormatError("GCT size must be <= 256")
        #Determine packed size
        while temp > 0:
            temp >>= 1
            packed_size += 1
        packed_byte |= ((packed_size - 1) & 7)
        
        #Recover the aspect ratio (what is this even FOR)
        if self._aspect:
            aspect_ratio = (self._aspect << 6) - 15
        else:
            aspect_ratio = 0
        
        #Pack the packed byte and other fields
        out += struct.pack("3B", packed_byte, self._bgcolor, aspect_ratio)
        
        #Pack the gct
        gct_size = 2 << (packed_size - 1)
        if gct_exists:
            #Pack entire GCT
            i = 0
            for color in self._gct:
                out += struct.pack('3B', *color)
                i += 1
            assert i <= gct_size
            #Pad with null colors
            while i < gct_size:
                out += b"\0\0\0"
                i += 1
                
        #Pack the ramaining blocks (these supply their own headers)
        for block in self._blocks:
            out += block.encode()
            
        #Pack the trailer
        out += GIF_FOOTER.to_bytes(1, "little")
        #Done
        return out
        
    #------------------------------------------------
    # Accessors
    #------------------------------------------------
    def images(self):
        '''Get all the image blocks'''
        return [b for b in self._blocks if isinstance(b, ImageBlock)]
        
    #------------------------------------------------
    # Mutation
    #------------------------------------------------
    def optimize(self):
        ''' Optimize the size of this image by removing extension blocks
            and resizing color tables.
        '''
        isize = len(self.encode())
        global_max = 0
        #Remove useless blocks
        i = 0
        while i < len(self._blocks):
            block = self._blocks[i]
            remove = False
            if isinstance(block, ExtensionBlock):
                #Optimize extensions
                if isinstance(block, GraphicExtension):
                    remove = not block.useful()
                else:
                    remove = True
            else:
                #Optimize local color tables
                if block._lct:
                    indices = lzw_decompress(
                        block._lzw, block._lzw_min, block._lct,
                    )
                    biggest = max(indices)
                    if biggest+1 < len(block._lct):
                        del block._lct[biggest+1:]
                else:
                    biggest = max(
                        lzw_decompress(block._lzw, block._lzw_min, self._gct)
                    )
                    if biggest > global_max:
                        global_max = biggest
            if remove:
                del self._blocks[i]
            else:
                i += 1
        #Optimize the global color table
        if global_max+1 < len(self._gct):
            del self._gct[global_max+1:]
        fsize = len(self.encode())
        return isize - fsize

#================================================================
# Testing
#================================================================
if __name__ == "__main__":

    from time import time

    def testGif(path):
        '''Unit test on one gif image'''
        print("Testing %s" % path)
        #Time the opening of the GIF image
        ti = time()
        g = Gif(path)
        tf = time()
        print("Opened in %f seconds" % (tf - ti))
        ti = tf
        #Time the decompression of the image block
        block = g.images()[0]
        d = list(block.decompress())
        tf = time()
        print("Decompressed in %f seconds" % (tf - ti))
        #Ensure the image block is actually valid
        assert len(d) == (g._height * g._width)
        ti = tf
        #Time the recompression of the image block
        block.compress(d)
        #Calculate differences in the image block
        tf = time()
        print("Compressed in %f seconds" % (tf - ti))
        #Uncompress the compress
        de = list(block.decompress())
        assert d == de
        #Done!
        print("Passed test!\n")

    testGif(r"dev\image\test\sample_1.gif")
    testGif(r"dev\image\test\writegif.gif")
    testGif(r"dev\image\test\animated.gif")
    testGif(r"dev\image\test\test.GIF")
    testGif(r"dev\image\test\audrey.gif")
    testGif(r"dev\image\test\audrey_big.gif")

    
