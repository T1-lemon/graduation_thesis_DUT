import { MenuItem, MenuItemProps, MenuList, MenuListProps } from '@mui/material'
import { styled } from '@mui/material/styles'

export const MenuItemOverride = styled(MenuItem)<MenuItemProps>(() => ({
  width: 200
}))

export const MenuListOverrride = styled(MenuList)<MenuListProps>(() => ({
  marginTop: 10
}))
