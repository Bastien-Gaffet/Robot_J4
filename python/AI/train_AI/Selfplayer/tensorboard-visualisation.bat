@echo off
echo Demarrage de TensorBoard...

start cmd /k "tensorboard --logdir=runs"

timeout /t 20 >nul

start http://localhost:6006/

echo Cette fenetre se fermera dans 5 secondes...
timeout /t 5 >nul
exit