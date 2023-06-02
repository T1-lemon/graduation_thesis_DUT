import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux'

import type { DispatchType, RootState } from './configStore'

type DispatchFunc = () => DispatchType
export const useAppDispatch: DispatchFunc = useDispatch
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector
