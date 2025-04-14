import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ImageSlider({ images, interval = 3000 }: { images: string[], interval?: number }) {
  const [index, setIndex] = useState(0);

  // Preload all images on mount.
  useEffect(() => {
    images.forEach((src) => {
      const img = new Image();
      img.src = src;
    });
  }, [images]);

  // Use functional updates to always get the latest state.
  const prevImage = () =>
    setIndex((prev) => (prev - 1 + images.length) % images.length);
  const nextImage = () =>
    setIndex((prev) => (prev + 1) % images.length);

  // Autoplay effect: auto-advance slides every `interval` milliseconds.
  useEffect(() => {
    const timer = setInterval(() => {
      nextImage();
    }, interval);
    return () => clearInterval(timer);
  }, [interval, images.length]);

  return (
    <div className="w-full max-w-4xl mx-auto text-center">
      <div className="relative">
        <AnimatePresence mode="wait">
          <motion.img
            key={images[index]}
            src={images[index]}
            alt={`Slide ${index + 1}`}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.3 }}
            className="w-full h-auto rounded-2xl shadow-md object-contain"
          />
        </AnimatePresence>
        <button
          onClick={prevImage}
          className="absolute top-1/2 left-2 transform -translate-y-1/2 bg-white dark:bg-zinc-800 p-2 rounded-full shadow hover:bg-zinc-100 dark:hover:bg-zinc-700"
          aria-label="Previous image"
        >
          ◀
        </button>
        <button
          onClick={nextImage}
          className="absolute top-1/2 right-2 transform -translate-y-1/2 bg-white dark:bg-zinc-800 p-2 rounded-full shadow hover:bg-zinc-100 dark:hover:bg-zinc-700"
          aria-label="Next image"
        >
          ▶
        </button>
      </div>
      <div className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">
        {index + 1} / {images.length}
      </div>
    </div>
  );
}
