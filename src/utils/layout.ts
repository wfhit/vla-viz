import { useEffect as useLayoutEffect, useState } from "react";

export interface ILayout {
    width: number;
    height: number;
    isDesktop: boolean;
    isPhone: boolean;
}

export function useScreenLayout() {
    let [layout, setLayout] = useState<ILayout>({ width: 0, height: 0, isDesktop: true, isPhone: false });

    useLayoutEffect(() => {
        // check the media queries that we use in css land
        let mediaQuery = window.matchMedia('screen and (max-width: 800px)');

        function handleResize() {
            setLayout({
                width: window.innerWidth,
                height: window.innerHeight,
                isDesktop: !mediaQuery.matches,
                isPhone: mediaQuery.matches,
            });
        }

        handleResize();

        window.addEventListener('resize', handleResize);
        mediaQuery.addEventListener('change', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
            mediaQuery.removeEventListener('change', handleResize);
        };
    }, []);

    return layout;
}
