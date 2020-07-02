#!/bin/bash
make -f make_serial.mk
make -f make_serialdecrypt.mk
make -f make_serial.mk clean
