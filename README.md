[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Build Status](https://travis-ci.org/Hasenpfote/fpq.svg?branch=master)](https://travis-ci.org/Hasenpfote/fpq)  

fpq
===

## About  
This package provides modules for manipulating floating point numbers quantization using NumPy.

## Feature
* Supports multidimensional arrays.  
* Supports encoding and decoding between 64/32/16-bits floating point numbers and N-bits unsigned normalized integers.  
* Supports encoding and decoding between 64/32/16-bits floating point numbers and N-bits signed normalized integers.  
* Supports encoding and decoding between 3d-vectors and N-bits unsigned integers.  
* Supports encoding and decoding between Quaternions and N-bits unsigned integers.  

## Compatibility  
* Python 3.3+

## Installation  
```
pip install git+https://github.com/Hasenpfote/fpq
```
or
```
python setup.py install
```

## Usage
[examples](example/)

Please refer to [the reference](https://hasenpfote.github.io/fpq/) for the details.

## References and links  
[D3D: Data Conversion Rules](https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx)  
[OGL: Normalized Integer](https://www.khronos.org/opengl/wiki/Normalized_Integer)  
[Vulkan: Fixed-Point Data Conversions](http://vulkan-spec-chunked.ahcox.com/ch02s08.html)

## License  
This software is released under the MIT License, see LICENSE.
