import { useRef, useState } from "react";

export type GalleryItem = { label: string; src: string; caption?: string };

/** Thumbnail grid; click opens a native <dialog> lightbox. */
export function Gallery({ items }: { items: GalleryItem[] }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [open, setOpen] = useState<GalleryItem | null>(null);
  if (!items.length) return <p className="muted small">No images.</p>;
  return (
    <>
      <div className="gallery">
        {items.map((it) => (
          <figure
            key={it.label + it.src.length}
            onClick={() => {
              setOpen(it);
              dialog.current?.showModal();
            }}
          >
            <img src={it.src} alt={it.label} loading="lazy" />
            <figcaption>{it.caption ?? it.label}</figcaption>
          </figure>
        ))}
      </div>
      <dialog ref={dialog} className="lightbox" onClick={() => dialog.current?.close()}>
        {open && (
          <>
            <img src={open.src} alt={open.label} />
            <p>{open.caption ?? open.label}</p>
          </>
        )}
      </dialog>
    </>
  );
}
