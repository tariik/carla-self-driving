#!/bin/bash
# Script simple para iniciar servidor CARLA sin GPU

echo "🚀 Iniciando servidor carla..."
echo ""

cd ~/carla

# Iniciar carla sin display (para servidores sin GPU)
DISPLAY= ./CarlaUE4.sh -carla-rpc-port=2000 -RenderOffScreen &

echo "Servidor carla iniciando en puerto 2000..."
echo "Espera 60 segundos para que cargue completamente"
echo ""
echo "Para detener: pkill -9 CarlaUE4"
