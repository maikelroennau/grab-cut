import argparse

import cv2
import numpy as np


class GrabCut:
    def __init__(self, image, output_path):
        self.original_image = cv2.imread(image)
        self.image = self.original_image.copy()
        self.segmented_image = np.zeros_like(self.original_image)
        self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)

        self.output_path = output_path

        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.rect = None

        self.drawing = False
        self.background = True
        self.segmentation_initialized = False


    def __del__(self):
        cv2.destroyAllWindows()


    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x = x
            self.y = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.image = self.original_image.copy()
                cv2.rectangle(self.image, (self.x, self.y), (x, y), [255, 0, 0], 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.w = abs(self.x - x)
            self.h = abs(self.y - y)
            self.x = min(self.x, x)
            self.y = min(self.y, y)
            self.image = self.original_image.copy()
            if self.x == x or self.y == y:
                cv2.rectangle(self.image, (self.x, self.y), (self.x + self.w, self.y + self.h), [255, 0, 0], 2)
            else:
                cv2.rectangle(self.image, (self.x, self.y), (x, y), [255, 0, 0], 2)
        cv2.imshow("Input", self.image)


    def draw_markers(self, event, x, y, flags, param):
        if self.background:
            color = [0, 0, 0]
            marker = 0
        else:
            color = [255, 255, 255]
            marker = 1

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.image, (x, y), 3, color, -1)
            cv2.circle(self.mask, (x, y), 3, marker, -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.image, (x, y), 3, color, -1)
                cv2.circle(self.mask, (x, y), 3, marker, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv2.circle(self.image, (x, y), 3, color, -1)
                cv2.circle(self.mask, (x, y), 3, marker, -1)
        cv2.imshow("Input", self.image)


    def segment(self, initialized=False):
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)

        if initialized:
            cv2.grabCut(self.original_image, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
        else:
            cv2.grabCut(self.original_image, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)


    def run(self):
        # Draw rectangle
        cv2.namedWindow("Input")
        cv2.setMouseCallback("Input", self.draw_rectangle)
        cv2.imshow("Input", self.image)

        print("\nGrabCut segmentation")
        print("\nCommands:")
        print("  - Esc        ==> Exit.")
        print("  - Left mouse ==> Draw rectangle or scribbles.")
        print("  - Space      ==> Switch between 'background' (default) and 'foreground' scribbles.")
        print("  - Enter      ==> Confirm rectangle and update segmentation.")
        print("  - s          ==> Save segmented image.")

        print("Instructions: \n")
        print("1. First draw a rectangle to delimit the segmentation area, then press 'Enter' to continue.")
        while True:
            key = cv2.waitKey(0)
            if key == 27 or key == 13:
                break

        if key == 13:
            # Segment and caputre drawings
            self.drawing = False
            self.rect = (self.x, self.y, self.w, self.h)

            if not self.segmentation_initialized:
                self.segmentation_initialized = True
                self.segment()
                mask = np.where((self.mask==1) + (self.mask==3), 255, 0).astype(np.uint8)
                self.segmented_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)

            cv2.namedWindow("Output")
            cv2.setMouseCallback("Input", self.draw_markers)

            print("2. Add scribbles as necessary. When done, press 'Enter' to update the segmentation. \n")
            while True:
                cv2.imshow("Input", self.image)
                cv2.imshow("Output", self.segmented_image)
                key = cv2.waitKey(0)

                if key == 27:
                    break
                elif key == 32:
                    if self.background:
                        self.background = False
                        print("Switched to 'foreground' scribbles'.")
                    else:
                        self.background = True
                        print("Switched to 'background' scribbles'.")
                    continue
                elif key == 13:
                    self.segment(self.segmentation_initialized)
                elif key == ord("s"):
                    if self.output_path.endswith(".png") or self.output_path.endswith(".jpg"):
                        output_path = self.output_path
                    else:
                        output_path = "output.jpg"
                    cv2.imwrite(output_path, self.segmented_image)
                    print(f"Segmented image saved to '{output_path}'.")

                mask = np.where((self.mask==1) + (self.mask==3), 255, 0).astype(np.uint8)
                self.segmented_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)


def main():
    parser = argparse.ArgumentParser(description="Segment images using the GrabCut Algorithm.")

    parser.add_argument(
        "-i",
        "--image",
        help="The image to segment.",
        required=True,
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Path where to save the segmented image.",
        default="segmented.png",
        type=str)

    args = parser.parse_args()

    grab_cut = GrabCut(args.image, args.output)
    grab_cut.run()


if __name__ == "__main__":
    main()
