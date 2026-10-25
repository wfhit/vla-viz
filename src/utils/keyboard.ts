import { createContext, useContext, useEffect } from "react";
import { Subscriptions, useFunctionRef } from "./hooks";
import { multiSortStableAsc } from "./array";

export enum KeyboardOrder {
    MainPage = 0,
}

interface IKeyHandler {
    order: KeyboardOrder;
    handler: (ev: KeyboardEvent) => void;
    receiveKeyUp: boolean;
}

interface IKeyHandlerOptions {
    isActive?: boolean;
    receiveKeyUp?: boolean;
}

class KeyboardManager {
    private handlers: IKeyHandler[] = [];
    public isFocused = false;
    public isFocusedSubs = new Subscriptions();

    constructor(
        public localFocus = false,
    ) {
        // if we're global, we're always receiving keyboard events
        // whereas if we're local, we only receive keyboard events when we're focused
        this.isFocused = !localFocus;
    }

    registerHandler(order: KeyboardOrder, handler: (ev: KeyboardEvent) => void, opts?: IKeyHandlerOptions): () => void {
        let newHandler: IKeyHandler = { order, handler, receiveKeyUp: opts?.receiveKeyUp ?? false };
        this.handlers.push(newHandler);
        return () => {
            this.handlers = this.handlers.filter(h => h !== newHandler);
        };
    }

    handleKey = (ev: KeyboardEvent) => {
        let handlersSorted = multiSortStableAsc(this.handlers, [a => a.order]);

        let propagationStopped = false;
        let oldStopPropagation = ev.stopPropagation;

        ev.stopPropagation = () => {
            propagationStopped = true;
            oldStopPropagation.call(ev);
        };

        for (let handler of handlersSorted) {
            if (ev.type === "keyup" && !handler.receiveKeyUp) {
                continue;
            }
            handler.handler(ev);
            if (propagationStopped) {
                break;
            }
        }
    }

    handleFocusInOut = (ev: FocusEvent) => {
        this.isFocused = ev.type === "focusin";
        this.isFocusedSubs.notify();
    }
}

export const KeyboardManagerContext = createContext<KeyboardManager>(new KeyboardManager());

export function useGlobalKeyboard(order: KeyboardOrder, handler: (ev: KeyboardEvent) => void, opts?: IKeyHandlerOptions): KeyboardManager {
    let manager = useContext(KeyboardManagerContext);
    let handlerRef = useFunctionRef(handler);
    let receiveKeyUp = opts?.receiveKeyUp ?? false;
    let isActive = opts?.isActive ?? true;

    useEffect(() => {
        if (isActive) {
            let h = (ev: KeyboardEvent) => handlerRef.current(ev);
            let unregister = manager.registerHandler(order, h, { receiveKeyUp });
            return () => unregister();
        }
    }, [order, handlerRef, manager, receiveKeyUp, isActive]);

    return manager;
}
