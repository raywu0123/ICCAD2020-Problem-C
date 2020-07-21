OUTPUT_DIR=$1
TEAM_ID="cada0048"

echo "Exporting GraphPreprocessor..."
pipenv run pyinstaller -F --distpath $OUTPUT_DIR GraphPreprocessing.py
rm GraphPreprocessing.spec
mv $OUTPUT_DIR/GraphPreprocessing $OUTPUT_DIR/"${TEAM_ID}_preprocessing"

echo "Exporting GPUSimulator..."
./scripts/build.sh
cp ./build/GPUSimulator $OUTPUT_DIR/"${TEAM_ID}_GPUsimulator"
