CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lzsibot -pthread
TARGET = zsibot_control

all: $(TARGET)

$(TARGET): main_sdk.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
